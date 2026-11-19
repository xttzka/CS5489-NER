"""Score NER predictions with entity-level P/R/F1 and sentence-bootstrap 95% intervals.

Several predictions are compared with a paired bootstrap; tags can first be projected to the three shared types.
"""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from eval.baseline_statistics import bootstrap_counts, compare, metric_values
from eval.result_tables import FIELDS, read_csv, validate_rows
from eval.score_ner import read_subset_ids, read_tags_by_id
from eval.score_transfer import MAPPING_PATH, project_tags


COUNTS = ("tp", "fp", "fn")
METRICS = ("precision", "recall", "f1")


def named_predictions(items):
    """Parse repeated NAME=PATH arguments into an ordered dict of paths."""
    predictions = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in predictions:
            raise ValueError(f"expected a unique NAME=PATH prediction: {item!r}")
        predictions[name] = Path(path)
    return predictions


def score_models(gold, predictions, *, types=None, resamples=1000, seed=42):
    """Compute exact-match metrics on aligned sentences, plus paired intervals."""
    if not predictions or not gold or resamples < 1:
        raise ValueError("expected nonempty data, predictions, and positive resamples")
    comparisons = {name: compare(gold, pred) for name, pred in predictions.items()}
    observed = {
        kind for _, per_sentence in comparisons.values()
        for record in per_sentence.values() for kind in record["per_type"]
    }
    types = tuple(sorted(observed)) if types is None else tuple(types)
    if len(types) != len(set(types)) or observed.difference(types):
        raise ValueError("types must be unique and include all observed entity types")
    names = tuple(predictions)
    ids = sorted(gold)
    type_index = {kind: index for index, kind in enumerate(types, start=1)}
    counts = np.zeros((len(ids), len(names), len(types) + 1, 3), dtype=np.int64)
    for model_index, name in enumerate(names):
        _, per_sentence = comparisons[name]
        for sentence_index, sample_id in enumerate(ids):
            record = per_sentence[sample_id]
            counts[sentence_index, model_index, 0] = [record[k] for k in COUNTS]
            for kind, values in record["per_type"].items():
                counts[sentence_index, model_index, type_index[kind]] = [values[k] for k in COUNTS]

    totals = counts.sum(axis=0)
    point = metric_values(totals)
    bootstrap = metric_values(bootstrap_counts(counts, resamples, seed))
    ci = np.quantile(bootstrap, (0.025, 0.975), axis=0)

    def result(model_index, group_index):
        return {
            **dict(zip(COUNTS, totals[model_index, group_index].tolist(), strict=True)),
            **dict(zip(METRICS, point[model_index, group_index].tolist(), strict=True)),
            "bootstrap_ci95": {
                metric: ci[:, model_index, group_index, metric_index].tolist()
                for metric_index, metric in enumerate(METRICS)
            },
        }

    models = {
        name: {
            "scores": result(index, 0),
            "per_type": {kind: result(index, group) for kind, group in type_index.items()},
        }
        for index, name in enumerate(names)
    }
    pairs = {}
    for first, second in combinations(range(len(names)), 2):
        diffs = bootstrap[:, second, 0, 2] - bootstrap[:, first, 0, 2]
        difference = float(point[second, 0, 2] - point[first, 0, 2])
        pairs[f"{names[second]}_minus_{names[first]}"] = {
            "first": names[first], "second": names[second],
            "f1_difference": difference,
            "ci95": np.quantile(diffs, (0.025, 0.975)).tolist(),
        }
    return {
        "sentence_count": len(ids), "entity_types": list(types),
        "models": models, "paired_f1_differences": pairs,
        "bootstrap": {"resamples": resamples, "seed": seed,
                      "unit": "sentence", "interval": "95% percentile"},
    }


def output_rows(result, args, source):
    """Yield one results-CSV row per method, entity type and metric, including interval bounds."""
    config = json.dumps({"bootstrap": result["bootstrap"], "scoring": args.scoring},
                        ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    label_space = "+".join(sorted(result["entity_types"]))
    for method, model in result["models"].items():
        for entity_type, scores in [("all", model["scores"]), *model["per_type"].items()]:
            base = {
                "member": "B", "experiment": args.experiment, "method": method,
                "condition": args.condition, "train_source": args.train_source,
                "dataset": args.dataset, "split": args.split, "scope": args.scope,
                "label_space": label_space, "entity_type": entity_type,
                "entity_group": "all", "seed": args.model_seed if args.model_seed is not None else "",
                "fold": "", "config_json": config,
            }
            prefix = f"{source}#/models/{method}/" + (
                "scores" if entity_type == "all" else f"per_type/{entity_type}"
            )
            for metric in METRICS:
                yield {**base, "metric": metric, "value": scores[metric], "unit": "ratio",
                       "source_ref": f"{prefix}/{metric}"}
                for position, boundary in enumerate(("lower", "upper")):
                    yield {**base, "metric": f"{metric}_ci95_{boundary}",
                           "value": scores["bootstrap_ci95"][metric][position], "unit": "ratio",
                           "source_ref": f"{prefix}/bootstrap_ci95/{metric}/{position}"}


def append_csv(path, new_rows):
    """Validate the new rows together with the existing CSV rows, then append them."""
    old = list(read_csv(path)) if path.exists() else []
    list(validate_rows([*old, *new_rows]))
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
        if new_file:
            writer.writeheader()
        writer.writerows(new_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--subset-ids", type=Path)
    parser.add_argument("--target-dataset", help="Gold schema for shared three-type scoring")
    parser.add_argument("--prediction-schema", help="Prediction schema for shared three-type scoring")
    parser.add_argument("--entity-types", nargs="+", help="Explicit native label schema, including types absent from this sample")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, help="Append model scores to this results CSV")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--condition", default="standard")
    parser.add_argument("--train-source", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--scope", default="")
    parser.add_argument("--model-seed", type=int)
    args = parser.parse_args()

    try:
        if args.output.exists():
            raise FileExistsError(f"{args.output} exists; choose a new output")
        if (args.target_dataset is None) != (args.prediction_schema is None):
            raise ValueError("target-dataset and prediction-schema must be supplied together")
        paths = named_predictions(args.pred)
        gold = read_tags_by_id(args.gold)
        if args.subset_ids:
            ids = read_subset_ids(args.subset_ids)
            missing = set(ids) - gold.keys()
            if missing:
                raise ValueError(f"subset ID {min(missing)!r} is absent from gold")
            gold = {sample_id: gold[sample_id] for sample_id in ids}
        predictions = {name: read_tags_by_id(path) for name, path in paths.items()}

        types = args.entity_types
        args.scoring = "native"
        if args.target_dataset:
            settings = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["transfer_three_types"]
            maps = settings["maps"]
            if maps.get(args.target_dataset) is None or maps.get(args.prediction_schema) is None:
                raise ValueError("no shared three-type mapping for one of the specified schemas")
            gold = project_tags(gold, maps[args.target_dataset])
            predictions = {name: project_tags(pred, maps[args.prediction_schema])
                           for name, pred in predictions.items()}
            if types is not None and set(types) != set(settings["types"]):
                raise ValueError("entity-types must match the shared transfer label schema")
            types = settings["types"]
            args.scoring = "transfer_three_types"

        result = score_models(gold, predictions, types=types,
                              resamples=args.resamples, seed=args.bootstrap_seed)
        result["inputs"] = {
            "gold": str(args.gold), "predictions": {name: str(path) for name, path in paths.items()},
            "subset_ids": str(args.subset_ids) if args.subset_ids else None,
            "scoring": args.scoring, "target_dataset": args.target_dataset,
            "prediction_schema": args.prediction_schema,
        }
        rows = list(output_rows(result, args, args.output)) if args.csv else []
        if args.csv:
            previous = list(read_csv(args.csv)) if args.csv.exists() else []
            list(validate_rows([*previous, *rows]))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2)
            target.write("\n")
        if args.csv:
            append_csv(args.csv, rows)
        for name, model in result["models"].items():
            score = model["scores"]
            print(f"{name}: P={score['precision']:.4f} R={score['recall']:.4f} "
                  f"F1={score['f1']:.4f} 95% CI={score['bootstrap_ci95']['f1']}")
        print(f"Report: {args.output}")
        if args.csv:
            print(f"Appended {len(rows)} rows: {args.csv}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
