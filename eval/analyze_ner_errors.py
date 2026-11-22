"""Sort the errors of an exact-match NER evaluation into five kinds."""

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import bio_to_spans, read_subset_ids, read_tags_by_id, score_ner


KINDS = ("missed", "spurious", "boundary", "type", "boundary_and_type")


def load_inputs(gold_paths, prediction_paths, subset_paths):
    gold, tokens, predictions = {}, {}, {}
    for path in gold_paths:
        records = read_tags_by_id(path)
        if gold.keys() & records.keys():
            raise ValueError(f"{path}: gold sample IDs repeat across input files")
        gold.update(records)
        with path.open(encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id, words = row["sample_id"], row.get("tokens")
                if not isinstance(words, list) or not all(isinstance(word, str) for word in words):
                    raise ValueError(f"{sample_id}: gold requires a list of tokens")
                if len(words) != len(gold[sample_id]):
                    raise ValueError(f"{sample_id}: gold token and tag counts differ")
                tokens[sample_id] = words
    for path in prediction_paths:
        records = read_tags_by_id(path)
        if predictions.keys() & records.keys():
            raise ValueError(f"{path}: prediction sample IDs repeat across input files")
        predictions.update(records)
    if subset_paths:
        ids = [sample_id for path in subset_paths for sample_id in read_subset_ids(path)]
        if len(ids) != len(set(ids)):
            raise ValueError("sample IDs repeat across subset lists")
        if set(ids) - gold.keys():
            raise ValueError(f"subset ID {min(set(ids) - gold.keys())!r} is missing from gold")
        gold = {sample_id: gold[sample_id] for sample_id in ids}
    return gold, predictions, tokens


def pair_overlaps(gold, predictions):
    """Pair sorted flat spans, preferring more pairs, same types, then more shared tokens."""
    n, m = len(gold), len(predictions)
    best = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    actions = [[None] * m for _ in range(n)]
    # BIO spans on each side are disjoint, so overlap pairs can be made in sentence order.
    for i in range(n - 1, -1, -1):
        start, end, kind = gold[i]
        for j in range(m - 1, -1, -1):
            pred_start, pred_end, pred_kind = predictions[j]
            overlap = min(end, pred_end) - max(start, pred_start)
            options = []
            if overlap > 0:
                pairs, same_type, shared_tokens = best[i + 1][j + 1]
                options.append(((pairs + 1, same_type + int(kind == pred_kind), shared_tokens + overlap), "pair"))
            options.extend(((best[i + 1][j], "skip_gold"), (best[i][j + 1], "skip_prediction")))
            best[i][j], actions[i][j] = max(options, key=lambda option: option[0])

    matches = []
    i = j = 0
    while i < n and j < m:
        action = actions[i][j]
        if action == "pair":
            matches.append((gold[i], predictions[j]))
        if action != "skip_prediction":
            i += 1
        if action != "skip_gold":
            j += 1
    return matches


def sentence_errors(gold, predictions):
    correct = gold & predictions
    unmatched_gold, unmatched_predictions = gold - correct, predictions - correct
    pairs = pair_overlaps(sorted(unmatched_gold), sorted(unmatched_predictions))
    errors = []
    for expected, predicted in pairs:
        if expected[:2] == predicted[:2]:
            category = "type"
        elif expected[2] == predicted[2]:
            category = "boundary"
        else:
            category = "boundary_and_type"
        errors.append((category, expected, predicted))
        unmatched_gold.remove(expected)
        unmatched_predictions.remove(predicted)
    errors.extend(("missed", span, None) for span in sorted(unmatched_gold))
    errors.extend(("spurious", None, span) for span in sorted(unmatched_predictions))
    return errors


def describe_span(span, tokens):
    if span is None:
        return None
    start, end, kind = span
    return {"start": start, "end": end, "type": kind, "text": " ".join(tokens[start:end])}


def analyze(gold, predictions, tokens):
    scores = score_ner(gold, predictions)
    errors = []
    for sample_id in sorted(gold):
        words = tokens[sample_id]
        spans = sentence_errors(bio_to_spans(gold[sample_id]), bio_to_spans(predictions[sample_id]))
        for index, (category, expected, predicted) in enumerate(spans, 1):
            errors.append({
                "error_id": f"{sample_id}:error:{index}",
                "sample_id": sample_id,
                "tokens": words,
                "category": category,
                "gold": describe_span(expected, words),
                "prediction": describe_span(predicted, words),
            })
    counts = Counter(row["category"] for row in errors)
    paired = counts["boundary"] + counts["type"] + counts["boundary_and_type"]
    if scores["fp"] != counts["spurious"] + paired or scores["fn"] != counts["missed"] + paired:
        raise ValueError("error totals do not match the exact-match scorer")
    return scores, errors, {category: counts[category] for category in KINDS}


def save_outputs(destination, errors, summary):
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination) as temporary:
        stage = Path(temporary)
        with (stage / "errors.jsonl").open("w", encoding="utf-8", newline="\n") as output:
            for row in errors:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name in ("errors.jsonl", "summary.json"):
            (stage / name).replace(destination / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, nargs="+", required=True, help="Processed gold JSONL file(s)")
    parser.add_argument("--pred", type=Path, nargs="+", required=True, help="Prediction JSONL file(s), same BIO labels as gold")
    parser.add_argument("--subset-ids", type=Path, nargs="+", help="Require exactly the IDs in these fixed lists")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for errors.jsonl and summary.json")
    args = parser.parse_args()
    try:
        gold, predictions, tokens = load_inputs(args.gold, args.pred, args.subset_ids)
        scores, errors, counts = analyze(gold, predictions, tokens)
        summary = {
            "gold": [path.as_posix() for path in args.gold],
            "predictions": [path.as_posix() for path in args.pred],
            "subset_ids": [path.as_posix() for path in args.subset_ids or ()],
            "exact_match": scores,
            "error_events": len(errors),
            "categories": counts,
            "pairing": "Remove exact matches, then maximize one-to-one overlaps within each sentence. "
                       "Prefer more same-type pairs, then more shared tokens.",
            "offsets": "Zero-based token offsets; end excluded.",
        }
        save_outputs(args.output_dir.resolve(), errors, summary)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"{scores['scored_records']:,} sentences | F1 {scores['f1']:.2%} | {len(errors):,} error events")
    print(" | ".join(f"{kind}: {count:,}" for kind, count in counts.items()))
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
