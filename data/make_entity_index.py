"""Freeze v1 entity indices for all six processed test datasets.

Each JSONL line is one gold entity. Its position is tokens[start:end] using
zero-based offsets, with end excluded. Match the entire token sequence against
CoNLL-2003 training entities with case ignored, regardless of entity type.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import bio_to_spans


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = DATA_DIR / "derived" / "entity_index"
DATASETS = ("conll2003", "conllpp", "wikigold", "btc", "wnut17", "mit_restaurant")
VERSION = "v1"


def read_entities(dataset, split):
    path = DATA_DIR / "processed" / dataset / f"{split}.jsonl"
    sample_ids = set()
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            sample_id = row["sample_id"]
            if row["dataset"] != dataset or row["split"] != split:
                raise ValueError(f"{path}:{line_number}: expected {dataset}/{split}")
            if not sample_id.startswith(f"{dataset}:{split}:") or sample_id in sample_ids:
                raise ValueError(f"{path}:{line_number}: invalid or repeated sample ID {sample_id!r}")
            sample_ids.add(sample_id)

            tokens, tags = row["tokens"], row["tags"]
            if len(tokens) != len(tags):
                raise ValueError(f"{sample_id}: tokens and tags differ in length")
            for start, end, entity_type in sorted(bio_to_spans(tags)):
                yield sample_id, start, end, entity_type, tokens[start:end]


def name_key(tokens):
    return tuple(token.casefold() for token in tokens)


def write_index(path, dataset, seen_names):
    total = seen = 0
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for sample_id, start, end, entity_type, tokens in read_entities(dataset, "test"):
            is_seen = name_key(tokens) in seen_names
            record = {
                "sample_id": sample_id,
                "start": start,
                "end": end,
                "text": " ".join(tokens),
                "entity_type": entity_type,
                "seen_in_training": is_seen,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            total += 1
            seen += is_seen
    if not total:
        raise ValueError(f"{dataset}: no test entities to index")
    return total, seen


def main():
    seen_names = {name_key(tokens) for _, _, _, _, tokens in read_entities("conll2003", "train")}
    if not seen_names:
        raise ValueError("CoNLL-2003 training has no reference entities")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT_ROOT) as staging:
        pending = []
        for dataset in DATASETS:
            output = OUTPUT_ROOT / dataset / f"test_{VERSION}.jsonl"
            staged = Path(staging) / f"{dataset}.jsonl"
            total, seen = write_index(staged, dataset, seen_names)
            if output.exists() and output.read_bytes() != staged.read_bytes():
                raise ValueError(f"{output}: frozen index differs; use a new version to change it")
            pending.append((dataset, staged, output, total, seen))

        for dataset, staged, output, total, seen in pending:
            action = "Verified"
            if not output.exists():
                output.parent.mkdir(parents=True, exist_ok=True)
                staged.replace(output)
                action = "Wrote"
            print(f"{dataset}: {total:,} entities, {seen:,} seen, {total - seen:,} unseen")
            print(f"{action} {output}")


if __name__ == "__main__":
    main()
