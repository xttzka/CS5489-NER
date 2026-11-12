"""Classify exact-match NER errors, sample them and summarize two independent reviews."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from eval.score_ner import bio_to_spans, read_subset_ids, read_tags_by_id, score_ner


KINDS = ("missed", "spurious", "boundary", "type", "boundary_and_type")
REVIEW_KINDS = ("type", "spurious")
JUDGMENTS = ("guideline_difference", "gold_error", "other")
REVIEW_FIELDS = (
    "error_id", "sample_id", "sentence", "automatic_category",
    "gold_text", "gold_type", "gold_start", "gold_end",
    "prediction_text", "prediction_type", "prediction_start", "prediction_end",
    "judgment", "notes",
)


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


def sample_errors(errors, kinds, size, seed):
    eligible = sorted((row for row in errors if row["category"] in kinds), key=lambda row: row["error_id"])
    return Random(seed).sample(eligible, min(size, len(eligible))), len(eligible)


def spreadsheet_text(value):
    # Keep dataset text inert when a review CSV is opened as a spreadsheet.
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def write_review(path, sample):
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for error in sample:
            row = {
                "error_id": error["error_id"], "sample_id": error["sample_id"],
                "sentence": spreadsheet_text(" ".join(error["tokens"])),
                "automatic_category": error["category"],
                "judgment": "", "notes": "",
            }
            for side in ("gold", "prediction"):
                span = error[side]
                if span is not None:
                    for field, value in span.items():
                        row[f"{side}_{field}"] = spreadsheet_text(value) if isinstance(value, str) else value
            writer.writerow(row)


def save_outputs(destination, errors, sample, summary):
    if destination.exists():
        raise FileExistsError(f"{destination}: choose a new output folder to preserve any reviewer edits")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / "output"
        stage.mkdir()
        with (stage / "errors.jsonl").open("w", encoding="utf-8", newline="\n") as output:
            for row in errors:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_review(stage / "review.csv", sample)
        stage.rename(destination)


def read_review(path):
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        fields = reader.fieldnames or []
        if not {"error_id", "judgment"}.issubset(fields) or len(set(fields)) != len(fields):
            raise ValueError(f"{path}: expected distinct CSV headers including error_id and judgment")
        for row in reader:
            location = f"{path}:{reader.line_num}"
            if None in row or None in row.values():
                raise ValueError(f"{location}: row does not match the CSV header")
            error_id = row["error_id"].strip()
            if not error_id or error_id in rows:
                raise ValueError(f"{location}: error_id must be nonempty and unique")
            judgment = row["judgment"].strip()
            if judgment not in JUDGMENTS:
                raise ValueError(f"{location}: judgment for {error_id!r} must be one of {', '.join(JUDGMENTS)}")
            rows[error_id] = {**row, "judgment": judgment}
    if not rows:
        raise ValueError(f"{path}: no completed review rows")
    return rows, set(fields)


def summarize_agreement(review_a, review_b):
    if review_a.samefile(review_b):
        raise ValueError("supply two distinct review CSV files")
    first, first_fields = read_review(review_a)
    second, second_fields = read_review(review_b)
    if first.keys() != second.keys():
        missing_in_a = len(second.keys() - first.keys())
        missing_in_b = len(first.keys() - second.keys())
        raise ValueError(f"review error IDs differ ({missing_in_a} missing in A, {missing_in_b} missing in B)")

    # Error IDs are numbered within a sentence. Separate model outputs can reuse an ID.
    shared_event_fields = [field for field in REVIEW_FIELDS[1:-2] if field in first_fields & second_fields]
    matrix = {label: dict.fromkeys(JUDGMENTS, 0) for label in JUDGMENTS}
    for error_id, row in first.items():
        peer = second[error_id]
        for field in shared_event_fields:
            if row[field] != peer[field]:
                raise ValueError(f"{error_id!r}: review sheets describe different events in column {field!r}")
        matrix[row["judgment"]][peer["judgment"]] += 1

    size = len(first)
    counts_a = {label: sum(matrix[label].values()) for label in JUDGMENTS}
    counts_b = {label: sum(row[label] for row in matrix.values()) for label in JUDGMENTS}
    consensus = {label: matrix[label][label] for label in JUDGMENTS}
    agreed = sum(consensus.values())
    chance_numerator = sum(counts_a[label] * counts_b[label] for label in JUDGMENTS)
    denominator = size * size - chance_numerator
    kappa = (agreed * size - chance_numerator) / denominator if denominator else None

    return {
        "reviewed": size,
        "judgment_options": JUDGMENTS,
        "reviewers": {
            reviewer: {"input": path.as_posix(), "judgment_counts": counts,
                       "guideline_difference_rate": counts["guideline_difference"] / size}
            for reviewer, path, counts in (("a", review_a, counts_a), ("b", review_b, counts_b))
        },
        "contingency_table_a_by_b": matrix,
        "observed_agreement": agreed / size,
        "expected_agreement_by_chance": chance_numerator / (size * size),
        "cohen_kappa": kappa,
        "cohen_kappa_status": "defined" if denominator else "undefined_same_single_label",
        "consensus": {
            "agreed": agreed, "disagreed": size - agreed, "judgment_counts": consensus,
            "guideline_difference_rate_among_agreed": consensus["guideline_difference"] / agreed if agreed else None,
        },
    }


def agreement_main(argv):
    parser = argparse.ArgumentParser(
        prog="python -m eval.analyze_ner_errors agreement",
        description="Summarize independent judgments of the same sampled errors. "
                    f"Allowed judgment values are {', '.join(JUDGMENTS)}.",
    )
    parser.add_argument("--review-a", type=Path, required=True, help="First completed review.csv copy")
    parser.add_argument("--review-b", type=Path, required=True, help="Second completed review.csv copy")
    parser.add_argument("--agreement-output", type=Path, required=True, help="New JSON file for agreement and rates")
    args = parser.parse_args(argv)
    try:
        summary = summarize_agreement(args.review_a, args.review_b)
        args.agreement_output.parent.mkdir(parents=True, exist_ok=True)
        with args.agreement_output.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(summary, indent=2) + "\n")
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Agreed on {summary['consensus']['agreed']} of {summary['reviewed']} reviewed errors")
    if summary["cohen_kappa"] is None:
        print("Cohen's kappa is undefined because both reviewers used the same single label.")
    else:
        print(f"Cohen's kappa: {summary['cohen_kappa']:.4f}")
    print(f"Agreement report: {args.agreement_output.resolve()}")


def main():
    if sys.argv[1:2] == ["agreement"]:
        agreement_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(
        description=__doc__, epilog="Run with 'agreement --help' to compare two completed review CSVs.",
    )
    parser.add_argument("--gold", type=Path, nargs="+", required=True, help="Processed gold JSONL file(s)")
    parser.add_argument("--pred", type=Path, nargs="+", required=True, help="Prediction JSONL file(s), same BIO labels as gold")
    parser.add_argument("--subset-ids", type=Path, nargs="+", help="Require exactly the IDs in these fixed lists")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for errors, summary and review sheet")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--sample-kinds", nargs="+", choices=KINDS, default=REVIEW_KINDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    try:
        gold, predictions, tokens = load_inputs(args.gold, args.pred, args.subset_ids)
        scores, errors, counts = analyze(gold, predictions, tokens)
        kinds = list(dict.fromkeys(args.sample_kinds))
        sample, eligible = sample_errors(errors, kinds, args.sample_size, args.seed)
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
            "review_sample": {
                "categories": kinds, "seed": args.seed, "requested": args.sample_size,
                "eligible": eligible, "sampled": len(sample),
                "sampled_by_category": dict(Counter(row["category"] for row in sample)),
                "judgment_options": list(JUDGMENTS),
            },
        }
        save_outputs(args.output_dir.resolve(), errors, sample, summary)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"{scores['scored_records']:,} sentences | F1 {scores['f1']:.2%} | {len(errors):,} error events")
    print(" | ".join(f"{kind}: {count:,}" for kind, count in counts.items()))
    print(f"Review sample: {len(sample):,} of {eligible:,} eligible events (seed {args.seed})")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
