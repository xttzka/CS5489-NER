"""Generate word-level BIO predictions from a saved BERT NER model."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from eval.score_ner import read_subset_ids
from models.neural.bert_baseline import fix_iob2
from models.neural.token_alignment import tokenize_with_unknown_fallback


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate project-format predictions from a BERT checkpoint."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the saved best_model directory.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL containing sample_id and tokens.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for sample_id and predicted tags.",
    )
    parser.add_argument(
        "--subset-ids",
        help="Optional text file specifying the exact sample IDs to predict.",
    )
    parser.add_argument("--batch-size", type=int, default=8)

    return parser.parse_args()


def read_rows(path):
    rows = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def main():
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    output_path = Path(args.output)

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. Choose another output path "
            "to avoid overwriting predictions."
        )

    rows = read_rows(args.input)
    by_id = {row["sample_id"]: row for row in rows}

    if len(by_id) != len(rows):
        raise ValueError("Input contains duplicate sample IDs.")

    if args.subset_ids:
        requested_ids = read_subset_ids(Path(args.subset_ids))
        missing_ids = [sample_id for sample_id in requested_ids if sample_id not in by_id]

        if missing_ids:
            raise ValueError(
                f"{len(missing_ids)} subset IDs are missing from the input. "
                f"Example: {missing_ids[0]}"
            )

        rows = [by_id[sample_id] for sample_id in requested_ids]

    if not rows:
        raise ValueError("No input sentences to predict.")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    id_to_label = {
        int(index): label
        for index, label in model.config.id2label.items()
    }

    predictions = []

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Sentences: {len(rows)}")

    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]

            _, encoding, affected = tokenize_with_unknown_fallback(batch_rows, tokenizer)
            if affected:
                print(json.dumps({"token_alignment_fallback": affected}, ensure_ascii=False), flush=True)

            if encoding["input_ids"].shape[1] > model.config.max_position_embeddings:
                raise ValueError(
                    "A tokenized sentence exceeds the model's maximum length. "
                    "The script will not silently truncate it."
                )

            positions_by_row = []

            for index, row in enumerate(batch_rows):
                word_ids = encoding.word_ids(batch_index=index)
                first_positions = []
                previous_word_id = None

                for position, word_id in enumerate(word_ids):
                    if word_id is None:
                        continue

                    if word_id != previous_word_id:
                        first_positions.append(position)

                    previous_word_id = word_id

                if len(first_positions) != len(row["tokens"]):
                    raise ValueError(
                        f'{row["sample_id"]}: token/subword alignment failed.'
                    )

                positions_by_row.append(first_positions)

            encoding = encoding.to(device)
            predicted_ids = model(**encoding).logits.argmax(dim=-1).cpu().tolist()

            for index, row in enumerate(batch_rows):
                tags = [
                    id_to_label[predicted_ids[index][position]]
                    for position in positions_by_row[index]
                ]

                predictions.append(
                    {
                        "sample_id": row["sample_id"],
                        "tags": fix_iob2(tags),
                    }
                )

            completed = min(start + args.batch_size, len(rows))

            if completed == len(rows) or completed % 400 == 0:
                print(f"Predicted {completed}/{len(rows)} sentences", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
