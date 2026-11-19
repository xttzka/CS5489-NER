"""BERT span classifier for the project's NER experiments."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from eval.score_ner import bio_to_spans, score_ner
from models.llm.common import TYPE_DESCRIPTIONS, load_rows, sample_train


MODEL_NAME = "bert-base-cased"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a BERT span classifier for NER."
    )

    parser.add_argument(
        "--dataset",
        default="conll2003",
        choices=["conll2003", "wnut17", "mit_restaurant"],
    )
    parser.add_argument(
        "--train-size",
        default="1.0",
        help='Sentence count such as "100", or proportion such as "1.0".',
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=0,
        help="Validation sentence count; 0 means the full validation set.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-span-length", type=int, default=14)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="output/span")

    return parser.parse_args()


class SpanClassifier(nn.Module):
    def __init__(self, num_entity_types, max_span_length):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        hidden_size = self.encoder.config.hidden_size
        length_size = 32

        self.length_embedding = nn.Embedding(
            max_span_length + 1,
            length_size,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2 + length_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_entity_types + 1),
        )

    def forward(self, encoding, batch_ids, starts, ends, lengths):
        hidden = self.encoder(**encoding).last_hidden_state

        start_vectors = hidden[batch_ids, starts]
        end_vectors = hidden[batch_ids, ends]
        length_vectors = self.length_embedding(lengths)

        span_vectors = torch.cat(
            [start_vectors, end_vectors, length_vectors],
            dim=-1,
        )

        return self.classifier(span_vectors)


def prepare_batch(
    rows,
    tokenizer,
    label_to_id,
    max_span_length,
    negative_ratio,
    rng,
    device,
    training,
):
    """Enumerate candidate spans; when training, label them and subsample the negatives."""

    encoding = tokenizer(
        [row["tokens"] for row in rows],
        is_split_into_words=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    if encoding["input_ids"].shape[1] > 512:
        raise ValueError(
            "A tokenized sentence exceeds 512 subwords; "
            "this script will not silently truncate it."
        )

    batch_ids = []
    start_positions = []
    end_positions = []
    lengths = []
    targets = []
    metadata = []

    for batch_id, row in enumerate(rows):
        word_count = len(row["tokens"])
        word_ids = encoding.word_ids(batch_index=batch_id)

        first_subword = [None] * word_count
        last_subword = [None] * word_count

        for position, word_id in enumerate(word_ids):
            if word_id is None:
                continue

            if first_subword[word_id] is None:
                first_subword[word_id] = position

            last_subword[word_id] = position

        if any(value is None for value in first_subword):
            raise ValueError(
                f'{row["sample_id"]}: word/subword alignment failed.'
            )

        candidates = []

        for start in range(word_count):
            final_end = min(word_count, start + max_span_length)

            for end in range(start + 1, final_end + 1):
                candidates.append((start, end))

        if training:
            gold = {
                (start, end): entity_type
                for start, end, entity_type in bio_to_spans(row["tags"])
            }

            too_long = [
                (start, end, entity_type)
                for (start, end), entity_type in gold.items()
                if end - start > max_span_length
            ]

            if too_long:
                raise ValueError(
                    f'{row["sample_id"]}: gold span exceeds '
                    f"max-span-length={max_span_length}: {too_long[0]}"
                )

            positive = [span for span in candidates if span in gold]
            negative = [span for span in candidates if span not in gold]

            # Avoid letting the much larger NONE class dominate training.
            negative_count = min(
                len(negative),
                max(32, len(positive) * negative_ratio),
            )

            selected = positive + rng.sample(negative, negative_count)
        else:
            gold = None
            selected = candidates

        for start, end in selected:
            batch_ids.append(batch_id)
            start_positions.append(first_subword[start])
            end_positions.append(last_subword[end - 1])
            lengths.append(end - start)
            metadata.append((batch_id, start, end))

            if training:
                entity_type = gold.get((start, end))
                targets.append(
                    0 if entity_type is None else label_to_id[entity_type]
                )

    tensors = {
        "batch_ids": torch.tensor(
            batch_ids, dtype=torch.long, device=device
        ),
        "starts": torch.tensor(
            start_positions, dtype=torch.long, device=device
        ),
        "ends": torch.tensor(
            end_positions, dtype=torch.long, device=device
        ),
        "lengths": torch.tensor(
            lengths, dtype=torch.long, device=device
        ),
    }

    target_tensor = None

    if training:
        target_tensor = torch.tensor(
            targets,
            dtype=torch.long,
            device=device,
        )

    return encoding.to(device), tensors, target_tensor, metadata


def spans_to_bio(word_count, proposals):
    """Resolve overlaps by confidence, then convert spans to IOB2 tags."""

    tags = ["O"] * word_count
    occupied = [False] * word_count

    # Highest confidence first. The remaining keys make ties deterministic.
    proposals = sorted(
        proposals,
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    )

    for confidence, start, end, entity_type in proposals:
        if any(occupied[start:end]):
            continue

        tags[start] = f"B-{entity_type}"

        for position in range(start + 1, end):
            tags[position] = f"I-{entity_type}"

        for position in range(start, end):
            occupied[position] = True

    return tags


def evaluate(
    model,
    loader,
    tokenizer,
    label_to_id,
    id_to_label,
    args,
    device,
):
    """Classify all candidate spans of every row and score the resulting tags."""
    model.eval()

    gold_by_id = {}
    predictions_by_id = {}

    with torch.no_grad():
        for rows in loader:
            encoding, tensors, _, metadata = prepare_batch(
                rows=rows,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                max_span_length=args.max_span_length,
                negative_ratio=args.negative_ratio,
                rng=None,
                device=device,
                training=False,
            )

            logits = model(encoding, **tensors)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted_ids = probabilities.max(dim=-1)

            confidence = confidence.cpu().tolist()
            predicted_ids = predicted_ids.cpu().tolist()

            proposals_by_row = [[] for _ in rows]

            for index, (batch_id, start, end) in enumerate(metadata):
                label_id = predicted_ids[index]

                if label_id == 0:
                    continue

                proposals_by_row[batch_id].append(
                    (
                        confidence[index],
                        start,
                        end,
                        id_to_label[label_id],
                    )
                )

            for batch_id, row in enumerate(rows):
                sample_id = row["sample_id"]

                predictions_by_id[sample_id] = spans_to_bio(
                    len(row["tokens"]),
                    proposals_by_row[batch_id],
                )

                gold_by_id[sample_id] = row["tags"]

    scores = score_ner(gold_by_id, predictions_by_id)
    return scores, predictions_by_id


def main():
    args = parse_args()

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.dev_size < 0
        or args.max_span_length < 1
        or args.negative_ratio < 1
    ):
        raise ValueError("Invalid training arguments.")

    output_dir = Path(args.output_dir)

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; choose a new output directory."
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    negative_rng = random.Random(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    entity_types = sorted(TYPE_DESCRIPTIONS[args.dataset])
    label_to_id = {
        entity_type: index + 1
        for index, entity_type in enumerate(entity_types)
    }
    id_to_label = {
        index: entity_type
        for entity_type, index in label_to_id.items()
    }

    train_rows = sample_train(
        args.dataset,
        args.train_size,
        args.seed,
    )
    dev_rows = load_rows(args.dataset, "validation")

    if args.dev_size:
        dev_rows = dev_rows[:args.dev_size]

    if not train_rows or not dev_rows:
        raise ValueError("Training and validation sets must not be empty.")

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Training sentences: {len(train_rows)}")
    print(f"Validation sentences: {len(dev_rows)}")
    print(f"Entity types: {entity_types}")
    print(f"Maximum span length: {args.max_span_length}")
    print("Loading tokenizer and pretrained BERT...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    model = SpanClassifier(
        num_entity_types=len(entity_types),
        max_span_length=args.max_span_length,
    ).to(device)

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
    loss_function = nn.CrossEntropyLoss()

    output_dir.mkdir(parents=True)

    config = vars(args).copy()
    config["model_name"] = MODEL_NAME
    config["device"] = str(device)
    config["entity_types"] = entity_types

    with (output_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_number, rows in enumerate(train_loader, start=1):
            encoding, tensors, targets, _ = prepare_batch(
                rows=rows,
                tokenizer=tokenizer,
                label_to_id=label_to_id,
                max_span_length=args.max_span_length,
                negative_ratio=args.negative_ratio,
                rng=negative_rng,
                device=device,
                training=True,
            )

            optimizer.zero_grad()

            logits = model(encoding, **tensors)
            loss = loss_function(logits, targets)

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

        scores, predictions = evaluate(
            model,
            dev_loader,
            tokenizer,
            label_to_id,
            id_to_label,
            args,
            device,
        )

        print(
            f"Epoch {epoch} finished | "
            f"mean loss: {epoch_loss / len(train_loader):.4f}"
        )
        print("Validation scores:")
        print(json.dumps(scores, indent=2))

        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "label_to_id": label_to_id,
                },
                output_dir / "best_model.pt",
            )

            tokenizer.save_pretrained(output_dir / "tokenizer")

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

            print(f"Best model and predictions saved to: {output_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
