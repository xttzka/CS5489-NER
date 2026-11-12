"""Extract entities by prompting an instruction-tuned LLM for a JSON list of entity phrases."""

import argparse
import json
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.score_ner import bio_to_spans
from models.llm.common import (
    EXAMPLE_SOURCE, GUIDELINES, PROJECT, RAW_OUTPUT, RESULTS, TYPE_DESCRIPTIONS, entities_to_tags,
    load_input, load_rows, per_type_scores, save_json, score_any, subset_path, write_predictions,
)


MODELS = {"1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "7b": "Qwen/Qwen2.5-7B-Instruct"}
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SYSTEM = "You extract named entities from text and answer with JSON only."


def gold_entities(row, types):
    """List the gold entities of a training sentence in reading order, limited to the target types."""
    spans = sorted(bio_to_spans(row["tags"]))
    return [{"text": " ".join(row["tokens"][start:end]), "type": kind} for start, end, kind in spans if kind in types]


def build_messages(text, descriptions, guidelines, examples):
    """Give the task once, then each example as a user sentence answered by its JSON list."""
    lines = ["Extract the named entities from each sentence.", "", "Entity types:"]
    lines += [f"- {kind}: {description}" for kind, description in descriptions.items()]
    if guidelines:
        lines += ["", "Annotation rules:"] + [f"- {rule}" for rule in guidelines]
    lines += [
        "",
        'Answer with one JSON list in the form [{"text": "...", "type": "..."}]. "text" is copied exactly '
        'from the sentence and "type" is one of the type names above. Answer [] when the sentence has no entity.',
        "",
    ]
    messages = [{"role": "system", "content": SYSTEM}]
    prefix = "\n".join(lines) + "\n"
    for sentence, entities in examples:
        messages.append({"role": "user", "content": f"{prefix}Sentence: {sentence}"})
        messages.append({"role": "assistant", "content": json.dumps(entities, ensure_ascii=False)})
        prefix = ""
    messages.append({"role": "user", "content": f"{prefix}Sentence: {text}"})
    return messages


def parse_entities(answer):
    """Read the first JSON list in the answer. Return None when no valid list of entities is found."""
    start = answer.find("[")
    if start < 0:
        return None
    try:
        # Text after the first complete list, such as a further made-up sentence, is ignored.
        items, _ = json.JSONDecoder().raw_decode(answer[start:])
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list):
        return None
    entities = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not isinstance(item.get("type"), str):
            return None
        entities.append({"text": item["text"], "type": item["type"]})
    return entities


class ExamplePicker:
    """Choose few-shot examples once at random, or per sentence by embedding similarity."""

    def __init__(self, pool, types, shots, selection, seed, device):
        self.shots, self.selection = shots, selection
        # Examples are training sentences with at least one entity of the target types.
        self.pool = [(" ".join(row["tokens"]), entities) for row in pool if (entities := gold_entities(row, types))]
        if shots and selection == "random":
            self.fixed = random.Random(seed).sample(self.pool, shots)
        if shots and selection == "similar":
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(ENCODER, device=device)
            self.vectors = self.encoder.encode(
                [text for text, _ in self.pool], batch_size=256, convert_to_tensor=True, normalize_embeddings=True,
            )

    def pick(self, texts):
        if not self.shots:
            return [[] for _ in texts]
        if self.selection == "random":
            return [self.fixed for _ in texts]
        queries = self.encoder.encode(texts, batch_size=256, convert_to_tensor=True, normalize_embeddings=True)
        nearest = torch.topk(queries @ self.vectors.T, self.shots, dim=1).indices.tolist()
        # The closest example sits last, right before the sentence to label.
        return [[self.pool[index] for index in reversed(row)] for row in nearest]


@torch.inference_mode()
def generate(model, tokenizer, prompts, batch_size, max_new_tokens):
    answers = []
    order = sorted(range(len(prompts)), key=lambda index: sum(len(turn["content"]) for turn in prompts[index]))
    for offset in range(0, len(order), batch_size):
        batch = order[offset:offset + batch_size]
        chats = [
            tokenizer.apply_chat_template(prompts[index], tokenize=False, add_generation_prompt=True)
            for index in batch
        ]
        inputs = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        texts = tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        answers.extend(zip(batch, texts, strict=True))
    return [text for _, text in sorted(answers)]


def run_dataset(args, model, tokenizer, dataset, scope, subset):
    schema = args.schema or dataset
    descriptions = TYPE_DESCRIPTIONS[schema]
    types = set(descriptions)
    rows = load_input(args.input) if args.input else load_rows(dataset, args.split, subset)
    texts = [" ".join(row["tokens"]) for row in rows]
    source = EXAMPLE_SOURCE.get(schema, schema)
    pool = load_rows(source, "train") if args.shots else []
    picker = ExamplePicker(pool, types, args.shots, args.selection, args.seed, model.device)
    guidelines = GUIDELINES.get(schema, ()) if args.guideline else ()
    if args.guideline and not guidelines:
        raise ValueError(f"{schema}: no annotation rules are written for this label scheme")
    prompts = [build_messages(text, descriptions, guidelines, examples)
               for text, examples in zip(texts, picker.pick(texts), strict=True)]

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    answers = generate(model, tokenizer, prompts, args.batch_size, args.max_new_tokens)
    parsed = [parse_entities(answer) for answer in answers]
    seconds = time.perf_counter() - start
    first_failures = [index for index, entities in enumerate(parsed) if entities is None]
    if first_failures:
        # One retry with a longer answer budget recovers lists cut off before the closing bracket.
        retried = generate(model, tokenizer, [prompts[index] for index in first_failures],
                           args.batch_size, 2 * args.max_new_tokens)
        for index, answer in zip(first_failures, retried, strict=True):
            answers[index], parsed[index] = answer, parse_entities(answer)
    failed = {rows[index]["sample_id"] for index, entities in enumerate(parsed) if entities is None}

    tags, totals = {}, {"entities": 0, "unknown_type": 0, "absent_from_sentence": 0}
    for row, entities in zip(rows, parsed, strict=True):
        tags[row["sample_id"]], counts = entities_to_tags(row["tokens"], entities or [], types)
        totals["entities"] += len(entities or [])
        for key, value in counts.items():
            totals[key] += value

    size = next(key for key, name in MODELS.items() if name == args.model_name)
    method = (("lora" if args.adapter else "qwen") + f"{size}_k{args.shots}"
              + (f"_{args.selection}" if args.shots else "") + ("_guide" if args.guideline else "_plain")
              + (f"_as_{schema}" if schema != dataset else "") + f"_{scope}")
    path = write_predictions(dataset, method, rows, tags, args.seed)
    raw_path = RAW_OUTPUT / dataset / method / f"{args.seed}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8", newline="\n") as target:
        for row, prompt, answer, entities in zip(rows, prompts, answers, parsed, strict=True):
            record = {"sample_id": row["sample_id"], "prompt": prompt, "answer": answer, "entities": entities}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    parsed_rows = [row for row in rows if row["sample_id"] not in failed]
    report = {
        "method": "LoRA-LLM" if args.adapter else "LLM prompt", "model": args.model_name, "adapter": args.adapter,
        "dataset": dataset, "split": None if args.input else args.split, "scope": scope,
        "label_scheme": schema, "input": args.input,
        "subset_ids": (None if subset is None or args.input
                       else subset_path(dataset, subset).relative_to(PROJECT).as_posix()),
        "shots": args.shots, "selection": args.selection if args.shots else None,
        "example_source": f"{source}:train" if args.shots else None,
        "guideline": args.guideline, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
        "type_descriptions": descriptions, "annotation_rules": list(guidelines),
        "predictions": path.relative_to(PROJECT).as_posix(),
        "scores": score_any(rows, tags, dataset, schema),
        "per_type": per_type_scores(rows, tags) if schema == dataset else None,
        "scores_on_parsed_sentences": (
            score_any(parsed_rows, {row["sample_id"]: tags[row["sample_id"]] for row in parsed_rows}, dataset, schema)
            if parsed_rows else None
        ),
        "output": {
            "sentences": len(rows),
            "parse_failures_first_attempt": len(first_failures),
            "parse_failures_after_retry": len(failed),
            "parse_failure_rate": len(failed) / len(rows),
            "entities": totals["entities"],
            "unknown_type": totals["unknown_type"],
            "absent_from_sentence": totals["absent_from_sentence"],
            "absent_from_sentence_rate": totals["absent_from_sentence"] / totals["entities"] if totals["entities"] else 0.0,
        },
        "timing": {
            "device": torch.cuda.get_device_name(0), "dtype": "bfloat16", "batch_size": args.batch_size,
            "inference_seconds": seconds, "inference_sentences_per_second": len(rows) / seconds,
            "gpu_vram_mb": torch.cuda.max_memory_allocated() / 2**20,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "includes": "tokenization, generation and JSON parsing of the first attempt",
        },
    }
    save_json(RESULTS / f"{method}_{dataset}_{args.split}_seed{args.seed}.json", report)
    output = report["output"]
    print(f"{dataset} {method}: F1={report['scores']['f1']:.2%} P={report['scores']['precision']:.2%} "
          f"R={report['scores']['recall']:.2%} | parse failures {output['parse_failures_after_retry']} "
          f"| absent {output['absent_from_sentence_rate']:.2%} | {report['timing']['inference_sentences_per_second']:.1f} sent/s",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="1.5b")
    parser.add_argument("--dataset", nargs="+", default=["conll2003"], choices=list(TYPE_DESCRIPTIONS))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--subset", default="validation_300", help="ID list name under data/subsets, or 'full'")
    parser.add_argument("--input", help="Derived test file to label instead of a split; needs --scope and one dataset")
    parser.add_argument("--scope", help="Name of the --input test set in method names and reports")
    parser.add_argument("--schema", help="Dataset whose labels the model outputs; the target dataset by default")
    parser.add_argument("--adapter", help="Directory of a LoRA adapter to load on the base model")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--selection", choices=("random", "similar"), default="random")
    parser.add_argument("--guideline", action="store_true", help="Add the annotation rules of the label scheme")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    args.model_name = MODELS[args.model]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if args.input and (not args.scope or len(args.dataset) != 1):
        parser.error("--input needs --scope and exactly one --dataset")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, PROJECT / args.adapter)
    model.eval()
    subset = None if args.subset == "full" else args.subset
    for dataset in args.dataset:
        run_dataset(args, model, tokenizer, dataset, args.scope or args.subset, subset)


if __name__ == "__main__":
    main()
