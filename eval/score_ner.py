"""Score exact entity matches in gold and prediction JSONL files."""

import argparse
import json
from pathlib import Path


def read_tags_by_id(path):
    """Read the same sample_id and tags fields from gold or prediction records."""
    records = {}
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{location}: expected one JSON object per line")
            sample_id = record.get("sample_id")
            tags = record.get("tags")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{location}: sample_id must be a nonempty string")
            if sample_id in records:
                raise ValueError(f"{location}: duplicate sample_id {sample_id!r}")
            if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                raise ValueError(f"{location}: tags must be a list of BIO strings")
            records[sample_id] = tags
    return records


def read_subset_ids(path):
    """Read one sample ID per line from a fixed evaluation list."""
    sample_ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if not sample_ids or any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"{path}: expected a nonempty sample_id on every line")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{path}: duplicate sample_id in the subset list")
    return sample_ids


def bio_to_spans(tags):
    """Convert strict IOB2 tags to (start, end, type) tuples, with end excluded."""
    spans = set()
    start = None
    entity_type = None

    # A final O closes an entity that extends to the last token.
    for index, tag in enumerate([*tags, "O"]):
        if tag.startswith("I-"):
            if start is None or tag[2:] != entity_type:
                raise ValueError(f"invalid IOB2 tag {tag!r} at token {index}")
            continue

        if start is not None:
            spans.add((start, index, entity_type))
            start = None
            entity_type = None

        if tag == "O":
            continue
        if not tag.startswith("B-") or not tag[2:].strip():
            raise ValueError(f"invalid BIO tag {tag!r} at token {index}")

        start = index
        entity_type = tag[2:]

    return spans


def score_ner(gold_by_id, pred_by_id, *, allow_subset=False):
    """Align records by ID and sum exact-match TP, FP, and FN across sentences."""
    if not gold_by_id or not pred_by_id:
        raise ValueError("gold and prediction files must both contain records")

    unknown_ids = pred_by_id.keys() - gold_by_id.keys()
    if unknown_ids:
        raise ValueError(f"{len(unknown_ids)} unknown prediction IDs, e.g. {min(unknown_ids)!r}")
    missing_ids = gold_by_id.keys() - pred_by_id.keys()
    if missing_ids and not allow_subset:
        raise ValueError(f"{len(missing_ids)} missing prediction IDs, e.g. {min(missing_ids)!r}")

    tp = fp = fn = 0
    for sample_id, pred_tags in pred_by_id.items():
        gold_tags = gold_by_id[sample_id]
        if len(gold_tags) != len(pred_tags):
            raise ValueError(
                f"{sample_id}: gold has {len(gold_tags)} tags, prediction has {len(pred_tags)}"
            )
        try:
            gold = bio_to_spans(gold_tags)
        except ValueError as error:
            raise ValueError(f"{sample_id} gold: {error}") from error
        try:
            pred = bio_to_spans(pred_tags)
        except ValueError as error:
            raise ValueError(f"{sample_id} prediction: {error}") from error

        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)

    return {
        "gold_records": len(gold_by_id),
        "scored_records": len(pred_by_id),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Processed dataset JSONL")
    parser.add_argument("--pred", type=Path, required=True, help="Model predictions JSONL")
    coverage = parser.add_mutually_exclusive_group()
    coverage.add_argument(
        "--allow-subset",
        action="store_true",
        help="Score just the IDs in the prediction file, e.g. a one-sentence demo",
    )
    coverage.add_argument(
        "--subset-ids",
        type=Path,
        help="Require predictions for exactly the IDs in this text file",
    )
    args = parser.parse_args()
    try:
        gold = read_tags_by_id(args.gold)
        if args.subset_ids:
            sample_ids = read_subset_ids(args.subset_ids)
            unknown_ids = set(sample_ids) - gold.keys()
            if unknown_ids:
                raise ValueError(f"subset ID {min(unknown_ids)!r} is absent from {args.gold}")
            gold = {sample_id: gold[sample_id] for sample_id in sample_ids}
        scores = score_ner(
            gold,
            read_tags_by_id(args.pred),
            allow_subset=args.allow_subset,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
