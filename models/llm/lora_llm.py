"""Fine-tune Qwen2.5-1.5B-Instruct with LoRA to answer the zero-shot extraction prompt with gold entity lists."""

import argparse
import json
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.llm.common import PROJECT, RESULTS, TYPE_DESCRIPTIONS, sample_train, save_json
from models.llm.prompt_llm import MODELS, build_messages, gold_entities


OUTPUT = PROJECT / "output" / "lora"


def encode(tokenizer, rows, descriptions):
    """Build chat inputs whose loss covers only the assistant's JSON answer."""
    inputs, labels = [], []
    for row in rows:
        messages = build_messages(" ".join(row["tokens"]), descriptions, (), [])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        answer = json.dumps(gold_entities(row, set(descriptions)), ensure_ascii=False) + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        inputs.append(prompt_ids + answer_ids)
        labels.append([-100] * len(prompt_ids) + answer_ids)
    width = max(map(len, inputs))
    pad = tokenizer.pad_token_id
    return (
        torch.tensor([[pad] * (width - len(ids)) + ids for ids in inputs]),
        torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in inputs]),
        torch.tensor([[-100] * (width - len(ids)) + ids for ids in labels]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="conll2003", choices=["conll2003", "wnut17", "mit_restaurant"])
    parser.add_argument("--train-size", default="0.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    name = MODELS["1.5b"]
    descriptions = TYPE_DESCRIPTIONS[args.dataset]
    training = sample_train(args.dataset, args.train_size, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda")
    config = LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, config)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    steps = args.epochs * -(-len(training) // args.batch_size)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0.0, 1 - step / steps))

    history = []
    order = list(range(len(training)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(order)
        losses = []
        for offset in range(0, len(order), args.batch_size):
            chunk = [training[index] for index in order[offset:offset + args.batch_size]]
            input_ids, mask, labels = (tensor.cuda() for tensor in encode(tokenizer, chunk, descriptions))
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        history.append({"epoch": epoch, "mean_loss": sum(losses) / len(losses)})
        print(f"epoch {epoch}/{args.epochs}: loss={history[-1]['mean_loss']:.4f}", flush=True)

    adapter = OUTPUT / f"qwen1.5b_{args.dataset}_n{args.train_size}_seed{args.seed}"
    model.save_pretrained(adapter)
    save_json(RESULTS / f"lora_train_{args.dataset}_n{args.train_size}_seed{args.seed}.json", {
        "method": "LoRA-LLM", "model": name, "train_source": args.dataset, "train_size": args.train_size,
        "training_sentences": len(training), "seed": args.seed, "epochs": args.epochs,
        "learning_rate": args.lr, "batch_size": args.batch_size, "rank": args.rank,
        "target_modules": config.target_modules if isinstance(config.target_modules, list) else sorted(config.target_modules),
        "prompt": "Zero-shot prompt with type descriptions and no annotation rules.",
        "adapter": adapter.relative_to(PROJECT).as_posix(), "history": history,
        "trainable_parameter_count": trainable, "parameter_count": total,
    })
    print(f"Adapter: {adapter}")


if __name__ == "__main__":
    main()
