"""Compare BERT and Span boundary errors with the team's overlap matching.

Boundary error fractions use paired error events (not FP+FN) as denominator.
An event with both a boundary and type error is reported separately and also
included in ``boundary_involved``. Perfect predictions have an undefined
fraction, represented by null; bootstrap ignores such replicates and reports
the number of usable replicates.
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from eval.analyze_ner_errors import KINDS, sentence_errors
from eval.baseline_statistics import bootstrap_counts
from eval.score_ner import bio_to_spans, read_subset_ids, read_tags_by_id, score_ner
from models.neural.score_b import named_predictions


def boundary_comparison(gold, predictions, resamples=1000, seed=42):
    if not gold or not predictions or resamples < 1:
        raise ValueError("nonempty gold/predictions and positive resamples are required")
    names = tuple(predictions)
    ids = sorted(gold)
    counts = np.zeros((len(ids), len(names), len(KINDS)), dtype=np.int64)
    exact = {}
    for j, name in enumerate(names):
        pred = predictions[name]
        exact[name] = score_ner(gold, pred)
        for i, sample_id in enumerate(ids):
            for kind, _, _ in sentence_errors(bio_to_spans(gold[sample_id]),
                                               bio_to_spans(pred[sample_id])):
                counts[i, j, KINDS.index(kind)] += 1
    totals = counts.sum(axis=0)
    sampled = bootstrap_counts(counts, resamples, seed)
    boundary = KINDS.index("boundary")
    both = KINDS.index("boundary_and_type")

    def fractions(matrix):
        denominator = matrix.sum(axis=-1)
        numerator = matrix[..., boundary] + matrix[..., both]
        return np.divide(numerator, denominator, out=np.full_like(denominator, np.nan, dtype=float),
                         where=denominator != 0)

    rates = fractions(totals)
    sampled_rates = fractions(sampled)

    def interval(values):
        usable = values[np.isfinite(values)]
        return {"ci95": np.quantile(usable, (0.025, 0.975)).tolist() if len(usable) else None,
                "valid_resamples": len(usable)}

    models = {}
    for i, name in enumerate(names):
        denominator = int(totals[i].sum())
        models[name] = {
            "exact_match": exact[name],
            "error_event_counts": dict(zip(KINDS, totals[i].tolist(), strict=True)),
            "total_error_events": denominator,
            "boundary_only_fraction_of_events": float(totals[i, boundary] / denominator) if denominator else None,
            "boundary_or_boundary_and_type_fraction_of_events": float(rates[i]) if denominator else None,
            "boundary_involved_bootstrap": interval(sampled_rates[:, i]),
        }
    comparisons = {}
    for i, j in combinations(range(len(names)), 2):
        differences = sampled_rates[:, j] - sampled_rates[:, i]
        comparisons[f"{names[j]}_minus_{names[i]}"] = {
            "boundary_involved_fraction_difference": (
                float(rates[j] - rates[i]) if np.isfinite(rates[i]) and np.isfinite(rates[j]) else None
            ),
            **interval(differences),
        }
    return {"sentence_count": len(ids), "models": models, "paired_differences": comparisons,
            "bootstrap": {"resamples": resamples, "seed": seed, "unit": "sentence",
                          "interval": "95% percentile"},
            "definition": "One-to-one overlapping-span error events; boundary_involved is boundary plus boundary_and_type divided by all error events."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--subset-ids", type=Path)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"{args.output} already exists")
        paths = named_predictions(args.pred)
        gold = read_tags_by_id(args.gold)
        if args.subset_ids:
            ids = read_subset_ids(args.subset_ids)
            missing = set(ids) - gold.keys()
            if missing:
                raise ValueError(f"subset ID {min(missing)!r} is absent from gold")
            gold = {sample_id: gold[sample_id] for sample_id in ids}
        predictions = {name: read_tags_by_id(path) for name, path in paths.items()}
        result = boundary_comparison(gold, predictions, args.resamples, args.seed)
        result["inputs"] = {"gold": str(args.gold), "predictions": {k: str(v) for k, v in paths.items()},
                            "subset_ids": str(args.subset_ids) if args.subset_ids else None}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
        for name, row in result["models"].items():
            print(f"{name}: errors={row['total_error_events']}; "
                  f"boundary-involved fraction={row['boundary_or_boundary_and_type_fraction_of_events']}")
        print(f"Report: {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
