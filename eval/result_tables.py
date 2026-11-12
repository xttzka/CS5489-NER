"""Export A's measured results and merge team CSVs with one shared schema."""

import argparse
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FIELDS = (
    "member", "experiment", "method", "condition", "train_source", "dataset", "split",
    "scope", "label_space", "entity_type", "entity_group", "seed", "fold",
    "metric", "value", "unit", "config_json", "source_ref",
)
KEY_FIELDS = tuple(field for field in FIELDS if field not in {"value", "unit", "source_ref"})


def numeric_leaves(node, path=()):
    """Yield numbers and their JSON paths, skipping descriptive text and Boolean flags."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from numeric_leaves(value, (*path, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from numeric_leaves(value, (*path, str(index)))
    elif type(node) in (int, float):
        yield path, node


def metric_unit(metric, path, value):
    if any(str(part).endswith(("_pp", "_percentage_points")) for part in path):
        return "percentage_point"
    if ("f1" in metric or "_fraction_" in metric or metric.endswith("_fraction")
            or metric.rsplit(".", 1)[-1] in {"precision", "recall", "token_accuracy", "overlap", "jaccard", "ari", "nmi"}
            or metric.startswith(("bootstrap_ci95.", "ci95."))):
        return "ratio"
    return "count" if type(value) is int else "number"


def report(name):
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8"))


def label_space(types):
    return "+".join(sorted(types))


def records(name, data, path=(), *, config=None, **context):
    node = data
    for part in path:
        node = node[int(part)] if isinstance(node, list) else node[part]
    for keys, value in numeric_leaves(node):
        dimensions = {"member": "A", "entity_type": "all", "entity_group": "all", **context}
        metric_parts = keys or (str(path[-1]),)
        if metric_parts[0] == "scores":
            metric_parts = metric_parts[1:]
        if metric_parts[0] == "per_type":
            dimensions["entity_type"] = metric_parts[1]
            metric_parts = metric_parts[2:]
        if metric_parts[-1] == "fold":
            continue
        metric = ".".join(metric_parts)
        unit = metric_unit(metric, (*path, *keys), value)
        yield {
            **dimensions, "metric": metric, "value": value, "unit": unit,
            "config_json": json.dumps(config or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "source_ref": f"results/{name}#/" + "/".join(
                str(part).replace("~", "~0").replace("/", "~1") for part in (*path, *keys)
            ),
        }


def baseline_rows():
    name = "m0_m2_baselines.json"
    data = report(name)
    for group_name, group in data["groups"].items():
        for dataset, target in group["targets"].items():
            for scope in ("full", "test_300"):
                if scope not in target:
                    continue
                base = ("groups", group_name, "targets", dataset, scope)
                context = dict(experiment="baseline", train_source=group["source_dataset"],
                               dataset=dataset, split="test", scope=scope,
                               label_space=label_space(target["types"]))
                for method in target[scope]["models"]:
                    features = group["m1_m2_features"] if method != "m0" else {}
                    yield from records(name, data, (*base, "models", method), method=method.upper(),
                                       condition="standard", seed=0, **context,
                                       config={"model": group[method], "features": features})
                for contrast in target[scope]["paired_f1_differences"]:
                    yield from records(name, data, (*base, "paired_f1_differences", contrast),
                                       method="paired", condition=contrast, **context,
                                       config=target[scope]["bootstrap"])
            for method in target["bio_repairs"]:
                yield from records(
                    name, data, ("groups", group_name, "targets", dataset, "bio_repairs", method),
                    experiment="bio_repairs", method=method.upper(), condition="standard",
                    train_source=group["source_dataset"], dataset=dataset, split="test",
                    scope=target["bio_repair_scope"], label_space=label_space(target["types"]), seed=0,
                )


def selection_rows():
    for name in ("m2_cv_grid.json", "m2_pos_cv_check.json", "m2_native_cv.json"):
        data = report(name)
        experiment = "regularization_cv_shared_pos" if name == "m2_pos_cv_check.json" else "regularization_cv"
        groups = [("conll2003", ())] if name != "m2_native_cv.json" else [
            (dataset, ("datasets", dataset)) for dataset in data["datasets"]
        ]
        for dataset, base in groups:
            selection = data if not base else data["datasets"][dataset]
            for section in ("candidates", "historical_outside_grid"):
                for index, candidate in enumerate(selection.get(section, ())):
                    path = (*base, section, str(index))
                    params = {key: candidate[key] for key in ("c1", "c2")}
                    context = dict(experiment=experiment, method="M2",
                                   condition=f"c1={params['c1']},c2={params['c2']}",
                                   train_source=dataset, dataset=dataset, split="train_cv",
                                   config={"parameters": params, "context_window": 1})
                    for metric in ("mean_f1", "std_f1"):
                        yield from records(name, data, (*path, metric), scope="cv_mean", **context)
                    for fold_index, fold in enumerate(candidate["fold_scores"]):
                        yield from records(name, data, (*path, "fold_scores", str(fold_index)),
                                           scope="cv_fold", fold=fold["fold"], **context)
            if "validation" in selection:
                for condition in selection["validation"]:
                    yield from records(name, data, (*base, "validation", condition),
                                       experiment="regularization_refit", method="M2", condition=condition,
                                       train_source=dataset, dataset=dataset, split="validation", scope="full")


def temporal_rows():
    name = "m0_m2_temporal.json"
    data = report(name)
    for dataset, methods in data["datasets"].items():
        for method in methods:
            for group in ("overall", "seen", "unseen"):
                yield from records(name, data, ("datasets", dataset, method, group),
                                   experiment="temporal", method=method.upper(), condition="standard",
                                   train_source="conll2003", dataset=dataset, split="test", scope="full",
                                   label_space=label_space(data["entity_types"]),
                                   entity_group="all" if group == "overall" else group, seed=0,
                                   config=data["models"][method])


def transfer_rows():
    name = "m2_pos_transfer.json"
    data = report(name)
    for dataset in data["targets"]:
        yield from records(
            name, data, ("targets", dataset), experiment="m2_pos_transfer",
            method="M2", condition="native_training", train_source=data["source_dataset"],
            dataset=dataset, split="test", scope="full", label_space=label_space(data["entity_types"]),
            config={"parameters": data["parameters"], "features": data["features"]},
        )

    name = "m2_transfer_matrix.json"
    data = report(name)
    for group, row in data["rows"].items():
        for dataset in row["targets"]:
            yield from records(
                name, data, ("rows", group, "targets", dataset), experiment="m2_transfer_matrix",
                method="M2", condition="shared_training", train_source="+".join(row["source_datasets"]),
                dataset=dataset, split="test", scope="full", label_space=label_space(data["entity_types"]),
                config={"parameters": data["parameters"], "features": data["features"]},
            )

    name = "m2_g2_transfer.json"
    data = report(name)
    for source, row in data["rows"].items():
        for dataset, target in row["targets"].items():
            base = ("rows", source, "targets", dataset)
            context = {
                "experiment": "m2_g2_transfer", "method": "M2", "train_source": source,
                "dataset": dataset, "split": "test", "scope": "full",
                "label_space": label_space(target["full"]["scores"]["per_type"]),
                "config": {key: data[key] for key in ("parameters", "pos_model", "use_chunk_tags")},
            }
            for condition in ("full", "without_G2"):
                yield from records(name, data, (*base, condition), condition=condition, **context)
            yield from records(name, data, (*base, "without_G2_minus_full_f1_pp"),
                               condition="without_G2_minus_full", **context)


def robustness_rows():
    name = "m0_m2_robustness.json"
    data = report(name)
    for experiment in ("replacement", "perturbations"):
        section = data[experiment]
        scope = f"{experiment}_{section['version']}"
        context = {
            "experiment": experiment, "train_source": "conll2003",
            "dataset": "conll2003", "split": "test", "scope": scope,
            "label_space": label_space(data["entity_types"]),
        }
        for method, config in data["models"].items():
            model = {**context, "method": method.upper(),
                     "config": {"model": config, "features": data["features"]}}
            for condition in section["conditions"]:
                path = (experiment, "conditions", condition, "models", method)
                yield from records(name, data, path, condition=condition, **model)
            if experiment == "replacement":
                path = (experiment, "replacement_targets", "models", method)
                yield from records(name, data, path, condition="replaced_entities", **model)

    for method in data["lowercase_vs_btc"]["models"]:
        for condition in ("original", "lowercase", "btc"):
            path = ("lowercase_vs_btc", "models", method, condition)
            yield from records(name, data, path, experiment="lowercase_vs_btc",
                               method=method.upper(), condition=condition, train_source="conll2003",
                               dataset="btc" if condition == "btc" else "conll2003",
                               split="test", scope="full",
                               label_space=label_space(data["lowercase_vs_btc"]["entity_types"]))


def analysis_rows():
    # Export aggregate sections explicitly. Keep raw entity names and individual answer rows local.
    sections = {
        "m2_feature_ablation.json": ("train_cv", "cv", ("fold_scores", "summary")),
        "m2_transfer_overlap.json": ("test", "full", ("all_non_source_targets", "by_source_non_source_targets", "points")),
        "m2_transfer_recall_groups.json": ("test", "full", ("datasets",)),
        "m2_seen_unseen_f1.json": ("test", "full", ("news_four_types", "transfer_three_types")),
        "m2_extra_features.json": ("test", "full", ("baseline", "conditions", "glove_type_coverage")),
        "m2_pos_validation.json": ("validation", "full", ("old_m2", "uniform_pos_m2", "f1_delta_percentage_points")),
        "m2_pos_casing.json": ("validation", "full", ("conditions", "lowercase_minus_original_pp")),
        "error_analysis/conll2003_m2/summary.json": ("test", "full", ("exact_match", "categories", "review_sample")),
    }
    for name, (split, scope, keys) in sections.items():
        data = report(name)
        context = dict(experiment=Path(name).stem if "/" not in name else "error_analysis",
                       method="M2", split=split, scope=scope,
                       config={key: data[key] for key in ("parameters", "features") if key in data})
        for section in keys:
            yield from records(name, data, (section,), condition=section, **context)


def exploration_rows():
    name = "data_exploration.json"
    data = report(name)
    context = {"experiment": "data_exploration", "method": "gold", "scope": "full"}
    for dataset, entry in data["datasets"].items():
        for split, stats in entry["splits"].items():
            yield from records(name, data, ("datasets", dataset, "splits", split),
                               condition="distribution", dataset=dataset, split=split,
                               label_space=label_space(stats["type_counts"]), **context)
    for left, targets in data["test_name_overlap"].items():
        for right in targets:
            yield from records(name, data, ("test_name_overlap", left, right),
                               condition="pairwise_name_overlap", train_source=left,
                               dataset=right, split="test", **context)
    yield from records(name, data, ("conll2003_train_test_shift",), condition="train_test_shift",
                       train_source="conll2003", dataset="conll2003", split="test", **context)


def ensemble_rows():
    name = "m0_m2_ensemble.json"
    data = report(name)
    context = dict(experiment="ensemble", train_source="conll2003", dataset="conll2003",
                   split="test", scope="test_300", label_space="LOC+MISC+ORG+PER")
    for group, result in data["groups"].items():
        yield from records(name, data, ("groups", group), method="Ensemble", condition=group,
                           config={"members": result["members"], "bootstrap": data["bootstrap"]}, **context)
    for index, pair in enumerate(data["pairwise_error_overlap"]):
        yield from records(name, data, ("pairwise_error_overlap", str(index)), method="paired",
                           condition="_vs_".join(pair["models"]), **context)


def clustering_rows():
    name = "entity_clustering_glove.json"
    data = report(name)
    for method in data["representations"]:
        yield from records(name, data, ("representations", method),
                           experiment="entity_clustering", method=method, condition=data["mode"],
                           dataset="conll2003", split="test", scope="full", label_space="LOC+MISC+ORG+PER",
                           config=data["cluster_config"])


def additional_rows():
    name = "m2_cv_controls.json"
    data = report(name)
    conll = dict(experiment="cv_controls", method="M2", train_source="conll2003",
                 dataset="conll2003", split="train_cv", scope="cv", config=data["parameters"])
    for key, value in data.items():
        if key == "conll2003_dev_check":
            yield from records(name, data, (key,), condition=key,
                               **{**conll, "split": "validation", "scope": "full"})
        elif key not in {"native_windows", "fingerprint", "parameters", "crf_options", "features"}:
            if isinstance(value, (dict, list)) or type(value) in (int, float):
                yield from records(name, data, (key,), condition=key, **conll)
    for dataset, result in data["native_windows"]["datasets"].items():
        native = {**conll, "train_source": dataset, "dataset": dataset, "config": result["parameters"]}
        yield from records(name, data, ("native_windows", "datasets", dataset),
                           condition="contiguous_windows", **native)


def validate_rows(rows):
    seen = set()
    for row in rows:
        if row.get("member") not in {"A", "B", "C"} or not all(row.get(key) for key in ("experiment", "method", "metric", "unit")):
            raise ValueError(f"missing member or result identifiers in row {row!r}")
        if not math.isfinite(float(row["value"])):
            raise ValueError("metric values must be finite numbers")
        config = json.loads(row.get("config_json") or "{}")
        if not isinstance(config, dict):
            raise ValueError("config_json must be a JSON object")
        normalized_config = json.dumps(config, sort_keys=True, separators=(",", ":"))
        key = tuple(normalized_config if field == "config_json" else str(row.get(field, "")) for field in KEY_FIELDS)
        if key in seen:
            raise ValueError(f"duplicate result for {key}")
        seen.add(key)
        yield row


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"{path}: header does not match the team schema")
        yield from reader


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        count = 0
        with staged.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            for row in validate_rows(rows):
                writer.writerow(row)
                count += 1
        staged.replace(path)
    print(f"{count:,} rows -> {path}")


def all_a_rows():
    yield from baseline_rows()
    yield from selection_rows()
    yield from temporal_rows()
    yield from transfer_rows()
    yield from robustness_rows()
    yield from analysis_rows()
    yield from exploration_rows()
    yield from ensemble_rows()
    yield from clustering_rows()
    yield from additional_rows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-a", help="Build A's CSV from completed JSON reports")
    export.add_argument("--output", type=Path, default=RESULTS / "A.csv")
    template = commands.add_parser("template", help="Print the shared CSV header")
    template.add_argument("--output", type=Path, help="Create a new UTF-8 CSV for member B or C")
    merge = commands.add_parser("merge", help="Validate and merge the specified member CSVs")
    merge.add_argument("--input", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "template":
            header = ",".join(FIELDS) + "\n"
            if args.output is None:
                print(header, end="")
            else:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8", newline="\n") as output:
                    output.write(header)
                print(args.output)
        elif args.command == "export-a":
            write_csv(args.output, all_a_rows())
        else:
            if args.output.resolve() in {path.resolve() for path in args.input}:
                raise ValueError("merge output must be a separate file")
            write_csv(args.output, (row for path in args.input for row in read_csv(path)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
