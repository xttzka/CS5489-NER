"""Build the M0 dictionary from training labels and predict on validation text."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import bio_to_spans


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed" / "conll2003"
DEFAULT_OUTPUT = PROJECT_DIR / "preds" / "conll2003" / "m0" / "0.jsonl"


def read_sentences(path, split):
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            if row["split"] != split:
                raise ValueError(f"{path}:{line_number}: expected the {split} split")
            yield row


def build_dictionary(training_sentences):
    """Keep a token sequence only when all its training entities have one type."""
    types_by_name = defaultdict(set)
    for row in training_sentences:
        tokens, tags = row["tokens"], row["tags"]
        if len(tokens) != len(tags):
            raise ValueError(f"{row['sample_id']}: token and tag counts differ")
        for start, end, entity_type in bio_to_spans(tags):
            name = tuple(tokens[start:end])
            types_by_name[name].add(entity_type)

    dictionary = {
        name: next(iter(types))
        for name, types in types_by_name.items()
        if len(types) == 1
    }
    return dictionary, len(types_by_name) - len(dictionary)


def predict_tags(tokens, dictionary, max_name_length):
    """Match exact tokens, prioritizing longer spans and then earlier starts."""
    candidates = []
    for start in range(len(tokens)):
        last_end = min(len(tokens), start + max_name_length)
        for end in range(start + 1, last_end + 1):
            entity_type = dictionary.get(tuple(tokens[start:end]))
            if entity_type is not None:
                candidates.append((start, end, entity_type))

    tags = ["O"] * len(tokens)
    for start, end, entity_type in sorted(candidates, key=lambda span: (-(span[1] - span[0]), span[0])):
        if any(tag != "O" for tag in tags[start:end]):
            continue
        tags[start] = f"B-{entity_type}"
        tags[start + 1:end] = [f"I-{entity_type}"] * (end - start - 1)
    return tags


def write_predictions(path, sentences, dictionary):
    path.parent.mkdir(parents=True, exist_ok=True)
    max_name_length = max(map(len, dictionary), default=0)
    count = 0
    with TemporaryDirectory(dir=path.parent) as staging:
        temporary_file = Path(staging) / path.name
        with temporary_file.open("w", encoding="utf-8", newline="\n") as target:
            for row in sentences:
                prediction = {
                    "sample_id": row["sample_id"],
                    "tags": predict_tags(row["tokens"], dictionary, max_name_length),
                }
                target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                count += 1
        if not count:
            raise ValueError("validation file contains no sentences")
        temporary_file.replace(path)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    parser.add_argument("--validation", type=Path, default=DATA_DIR / "validation.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    if output in {args.train.resolve(), args.validation.resolve()}:
        parser.error("--output must differ from the training and validation files")

    try:
        dictionary, ambiguous = build_dictionary(read_sentences(args.train, "train"))
        count = write_predictions(output, read_sentences(args.validation, "validation"), dictionary)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")

    print(f"Dictionary: {len(dictionary):,} names kept, {ambiguous:,} ambiguous names excluded")
    print(f"Wrote {count:,} validation predictions to {output}")


if __name__ == "__main__":
    main()
