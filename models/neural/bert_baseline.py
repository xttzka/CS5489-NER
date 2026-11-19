"""Train and evaluate a BERT token classifier on the project's JSONL data."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from eval.score_ner import score_ner
from models.llm.common import load_rows, sample_train


MODEL_NAME = "bert-base-cased"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BERT NER baseline and evaluate on validation data."
    )

    parser.add_argument(
        "--dataset",
        default="conll2003",
        choices=["conll2003", "wnut17", "mit_restaurant"],
    )
    parser.add_argument(
        "--train-size",
        default="100",
        help='Sentence count, e.g. "100", or proportion, e.g. "0.1" or "1.0".',
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=64,
        help="Number of validation sentences. Use 0 for the full validation set.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="output/bert_smoke",
        help="Directory for the model, validation predictions and configuration.",
    )

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fix_iob2(tags):
    """Convert an illegal I-X at the start of an entity into B-X."""
    fixed = []

    for tag in tags:
        if tag.startswith("I-"):
            entity_type = tag[2:]
            valid_previous = {f"B-{entity_type}", f"I-{entity_type}"}

            if not fixed or fixed[-1] not in valid_previous:
                tag = f"B-{entity_type}"

        fixed.append(tag)

    return fixed


def encode_batch(rows, tokenizer, label_to_id, max_positions):
    """Tokenize a batch and align word-level labels with first subwords."""
    encoding = tokenizer(
        [row["tokens"] for row in rows],
        is_split_into_words=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    sequence_length = encoding["input_ids"].shape[1]

    if sequence_length > max_positions:
        raise ValueError(
            f"Tokenized sequence length {sequence_length} exceeds "
            f"the model limit {max_positions}. Do not silently truncate."
        )

    labels = torch.full_like(encoding["input_ids"], -100)
    first_positions = []

    for batch_index, row in enumerate(rows):
        word_ids = encoding.word_ids(batch_index=batch_index)
        positions = []
        previous_word_id = None

        for position, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            if word_id != previous_word_id:
                positions.append(position)
                labels[batch_index, position] = label_to_id[
                    row["tags"][word_id]
                ]

            previous_word_id = word_id

        if len(positions) != len(row["tokens"]):
            raise ValueError(
                f'{row["sample_id"]}: token/subword alignment failed: '
                f'{len(positions)} positions for {len(row["tokens"])} words'
            )

        first_positions.append(positions)

    return encoding, labels, first_positions


def evaluate(model, loader, tokenizer, label_to_id, id_to_label, device):
    model.eval()
    gold_by_id = {}
    predictions_by_id = {}

    with torch.no_grad():
        for rows in loader:
            encoding, _, first_positions = encode_batch(
                rows,
                tokenizer,
                label_to_id,
                model.config.max_position_embeddings,
            )

            encoding = encoding.to(device)
            logits = model(**encoding).logits
            predicted_ids = logits.argmax(dim=-1).cpu().tolist()

            for index, row in enumerate(rows):
                tags = [
                    id_to_label[predicted_ids[index][position]]
                    for position in first_positions[index]
                ]

                gold_by_id[row["sample_id"]] = row["tags"]
                predictions_by_id[row["sample_id"]] = fix_iob2(tags)

    scores = score_ner(gold_by_id, predictions_by_id)
    return scores, predictions_by_id


def main():
    args = parse_args()

    if (
        args.epochs < 1
        or args.eval_every < 1
        or args.batch_size < 1
        or args.dev_size < 0
    ):
        raise ValueError(
            "epochs, eval-every and batch-size must be positive; "
            "dev-size must be >= 0"
        )
    set_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Build the label vocabulary from the full training split, even when the
    # actual training run uses only a small sample.
    full_train_rows = load_rows(args.dataset, "train")
    entity_types = sorted(
        {
            tag[2:]
            for row in full_train_rows
            for tag in row["tags"]
            if tag != "O"
        }
    )

    labels = ["O"]
    for entity_type in entity_types:
        labels.extend([f"B-{entity_type}", f"I-{entity_type}"])

    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    # This sampler is shared with member C's few-shot example pools.
    train_rows = sample_train(args.dataset, args.train_size, args.seed)
    dev_rows = load_rows(args.dataset, "validation")

    if args.dev_size:
        dev_rows = dev_rows[:args.dev_size]

    if not train_rows or not dev_rows:
        raise ValueError("Training and validation sets must not be empty.")

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Training sentences: {len(train_rows)}")
    print(f"Validation sentences: {len(dev_rows)}")
    print(f"Labels: {labels}")
    print("Loading tokenizer and pretrained BERT; first run may download files...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(device)

    train_loader = DataLoader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=list,
    )
    dev_loader = DataLoader(
        dev_rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=list,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["model_name"] = MODEL_NAME
    config["device"] = str(device)
    config["labels"] = labels

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        started = time.time()

        for batch_number, rows in enumerate(train_loader, start=1):
            encoding, aligned_labels, _ = encode_batch(
                rows,
                tokenizer,
                label_to_id,
                model.config.max_position_embeddings,
            )

            encoding = encoding.to(device)
            aligned_labels = aligned_labels.to(device)

            optimizer.zero_grad()
            outputs = model(**encoding, labels=aligned_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_number == 1 or batch_number % 10 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"batch {batch_number}/{len(train_loader)} | "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

        if epoch % args.eval_every != 0 and epoch != args.epochs:
            print(
                f"Epoch {epoch} finished in {time.time() - started:.1f}s | "
                f"mean loss: {epoch_loss / len(train_loader):.4f} | "
                "validation skipped"
            )
            continue

        scores, predictions = evaluate(
            model,
            dev_loader,
            tokenizer,
            label_to_id,
            id_to_label,
            device,
        )

        print(
            f"Epoch {epoch} finished in {time.time() - started:.1f}s | "
            f"mean loss: {epoch_loss / len(train_loader):.4f}"
        )
        print("Validation scores:")
        print(json.dumps(scores, indent=2))

        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]

            model.save_pretrained(output_dir / "best_model")
            tokenizer.save_pretrained(output_dir / "best_model")

            with (output_dir / "validation_scores.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(scores, file, indent=2)

            with (output_dir / "validation_predictions.jsonl").open(
                "w", encoding="utf-8"
            ) as file:
                for sample_id, tags in predictions.items():
                    record = {"sample_id": sample_id, "tags": tags}
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Best checkpoint and validation predictions saved to: {output_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
