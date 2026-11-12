"""Score strict span-majority NER ensembles and pairwise model error overlap."""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from eval.baseline_statistics import COUNTS, bootstrap_counts, compare, metric_values
from eval.score_ner import bio_to_spans, read_subset_ids, read_tags_by_id


RESAMPLES = 1_000
SEED = 42


def scoped_predictions(name, predictions, gold, selected_ids):
    unknown = predictions.keys() - gold.keys()
    if unknown:
        raise ValueError(f"{name}: unknown prediction ID {min(unknown)!r}")
    missing = set(selected_ids) - predictions.keys()
    if missing:
        raise ValueError(f"{name}: missing prediction for {min(missing)!r} in the evaluation scope")
    return {sample_id: predictions[sample_id] for sample_id in selected_ids}


def majority_answers(gold, predictions, members, votes_required):
    answers = {}
    for sample_id, gold_tags in gold.items():
        counts = Counter(span for name in members for span in predictions[name][sample_id])
        accepted = sorted(span for span, votes in counts.items() if votes >= votes_required)
        tags = ["O"] * len(gold_tags)
        end_of_previous = 0
        for start, end, kind in accepted:
            # Two overlapping spans cannot both have a strict majority of valid BIO answers.
            if start < end_of_previous:
                raise ValueError(f"{sample_id}: majority spans overlap")
            tags[start:end] = [f"B-{kind}"] + [f"I-{kind}"] * (end - start - 1)
            end_of_previous = end
        answers[sample_id] = tags
    return answers


def pairwise_error_overlap(gold, predictions):
    gold_set = {(sample_id, *span) for sample_id, spans in gold.items() for span in spans}
    errors = {}
    for model, answers in predictions.items():
        predicted = {(sample_id, *span) for sample_id, spans in answers.items() for span in spans}
        missed, spurious = gold_set - predicted, predicted - gold_set
        errors[model] = {"missed": missed, "spurious": spurious, "combined": missed | spurious}

    overlaps = []
    for left, right in combinations(errors, 2):
        result = {"models": [left, right]}
        for kind, left_errors in errors[left].items():
            right_errors = errors[right][kind]
            intersection = len(left_errors & right_errors)
            union = len(left_errors | right_errors)
            result[kind] = {
                "left_count": len(left_errors), "right_count": len(right_errors),
                "intersection": intersection, "union": union,
                "jaccard": intersection / union if union else None,
            }
        overlaps.append(result)
    return overlaps


def paired_f1_ci95_pp(best_counts, ensemble_counts):
    counts = np.array([
        [[rows[sample_id][key] for key in COUNTS] for rows in (best_counts, ensemble_counts)]
        for sample_id in sorted(best_counts)
    ], dtype=np.int64)
    scores = metric_values(bootstrap_counts(counts, RESAMPLES, SEED))[:, :, 2]
    return (100 * np.quantile(scores[:, 1] - scores[:, 0], (0.025, 0.975))).tolist()


def evaluate(gold, answers, groups):
    comparisons = {}
    for name, tags in answers.items():
        try:
            comparisons[name] = compare(gold, tags)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error

    gold_spans = {sample_id: bio_to_spans(tags) for sample_id, tags in gold.items()}
    predictions = {
        name: {sample_id: bio_to_spans(tags) for sample_id, tags in rows.items()}
        for name, rows in answers.items()
    }
    model_scores = {name: scores for name, (scores, _) in comparisons.items()}
    results = {}
    for group, members in groups.items():
        unknown = set(members) - answers.keys()
        if unknown:
            raise ValueError(f"{group}: unknown ensemble member {min(unknown)!r}")
        if len(members) < 2 or len(set(members)) != len(members):
            raise ValueError(f"{group}: expected at least two distinct model names")

        votes_required = len(members) // 2 + 1
        ensemble = majority_answers(gold, predictions, members, votes_required)
        scores, counts = compare(gold, ensemble)
        best = max(sorted(members), key=lambda name: model_scores[name]["f1"])
        best_scores, best_counts = comparisons[best]
        results[group] = {
            "members": members,
            "votes_required": votes_required,
            "scores": scores,
            "best_constituent": {"model": best, "f1": best_scores["f1"]},
            "f1_delta_vs_best_pp": 100 * (scores["f1"] - best_scores["f1"]),
            "paired_f1_delta_ci95_pp": paired_f1_ci95_pp(best_counts, counts),
        }
    return {
        "models": model_scores,
        "groups": results,
        "pairwise_error_overlap": pairwise_error_overlap(gold_spans, predictions),
    }


def assignment(value):
    name, separator, item = value.partition("=")
    if not separator or not name.strip() or not item.strip() or "," in name:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return name.strip(), item.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, type=Path, help="Full gold JSONL")
    parser.add_argument("--subset-ids", type=Path, help="Evaluate these gold IDs; other known IDs are ignored")
    parser.add_argument("--pred", required=True, action="append", type=assignment, metavar="NAME=PATH")
    parser.add_argument("--group", action="append", type=assignment, metavar="NAME=MODEL,MODEL,...",
                        help="Repeat for multiple ensembles; default is a majority of all supplied models")
    parser.add_argument("--output", type=Path, help="Write report JSON here; default prints JSON")
    args = parser.parse_args()

    paths = {name: Path(path) for name, path in args.pred}
    groups = {name: [model.strip() for model in value.split(",")] for name, value in args.group or []}
    if len(paths) != len(args.pred) or len(groups) != len(args.group or []):
        parser.error("model and group names must each be unique")
    if not groups:
        groups = {"majority": list(paths)}
    if args.output and args.output.resolve() in {
        path.resolve() for path in (args.gold, args.subset_ids, *paths.values()) if path is not None
    }:
        parser.error("--output must differ from all inputs")

    try:
        gold = read_tags_by_id(args.gold)
        ids = read_subset_ids(args.subset_ids) if args.subset_ids else list(gold)
        unknown = set(ids) - gold.keys()
        if unknown:
            raise ValueError(f"subset ID {min(unknown)!r} is absent from the gold file")
        answers = {name: scoped_predictions(name, read_tags_by_id(path), gold, ids)
                   for name, path in paths.items()}
        report = {
            "gold": args.gold.as_posix(),
            "subset_ids": args.subset_ids.as_posix() if args.subset_ids else None,
            "scored_records": len(ids),
            "prediction_files": {name: path.as_posix() for name, path in paths.items()},
            "vote_rule": "Strict majority for an identical token span and entity type.",
            "best_constituent_rule": "Largest observed F1 in this scope; ties favor the alphabetically first name.",
            "error_overlap_rule": "Jaccard of exact typed span errors by sample ID; null for an empty union.",
            "bootstrap": {
                "resamples": RESAMPLES, "seed": SEED, "unit": "sentence", "interval": "95% percentile",
                "comparison": "Ensemble minus the fixed best observed constituent in each group.",
            },
            **evaluate({sample_id: gold[sample_id] for sample_id in ids}, answers, groups),
        }
        content = json.dumps(report, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=args.output.parent) as staging:
                staged = Path(staging) / args.output.name
                staged.write_text(content, encoding="utf-8")
                staged.replace(args.output)
            print(f"Results: {args.output}")
        else:
            print(content, end="")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
