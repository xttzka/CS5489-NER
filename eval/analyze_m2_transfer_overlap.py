"""Relate common-type entity-name overlap to the M2 transfer F1 difference."""

import json
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from scipy.stats import linregress, spearmanr

from data.make_entity_index import DATA_DIR, name_key, read_entities
from eval.score_transfer import MAPPING_PATH


PROJECT = DATA_DIR.parent
MATRIX = PROJECT / "results" / "m2_transfer_matrix.json"
RESULTS = PROJECT / "results" / "m2_transfer_overlap.json"


def shared_entities(dataset, split, maps):
    for _, _, _, native_type, tokens in read_entities(dataset, split):
        entity_type = maps[dataset][native_type]
        if entity_type is not None:
            yield entity_type, name_key(tokens)


def describe_relationship(points):
    overlap = [point["overlap"] * 100 for point in points]
    loss = [point["f1_loss_pp"] for point in points]
    fit = linregress(overlap, loss)
    return {
        "points": len(points),
        "pearson_r": float(fit.rvalue),
        "spearman_rho": float(spearmanr(overlap, loss).statistic),
        "ols_slope_f1_loss_pp_per_overlap_pp": float(fit.slope),
        "ols_intercept_pp": float(fit.intercept),
        "ols_r_squared": float(fit.rvalue**2),
    }


def main():
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))["rows"]
    settings = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["transfer_three_types"]
    kinds, maps = settings["types"], settings["maps"]
    test_names = {
        dataset: list(shared_entities(dataset, "test", maps)) for dataset in matrix["conll2003"]["targets"]
    }
    training_names = {
        dataset: set(shared_entities(dataset, "train", maps)) for dataset in ("conll2003", "wnut17")
    }

    points, by_source = [], {}
    for source, row in matrix.items():
        source_datasets = row["source_datasets"]
        known = set().union(*(training_names[dataset] for dataset in source_datasets))
        baseline = {
            kind: mean(row["targets"][dataset]["scores"]["per_type"][kind]["f1"] for dataset in source_datasets)
            for kind in kinds
        }
        source_points = []
        for target, result in row["targets"].items():
            for kind in kinds:
                names = [name for entity_type, name in test_names[target] if entity_type == kind]
                scores = result["scores"]["per_type"][kind]
                if len(names) != scores["tp"] + scores["fn"]:
                    raise ValueError(f"{source}/{target}/{kind}: target entity count differs from matrix")
                matches = sum((kind, name) in known for name in names)
                source_points.append({
                    "source": source, "target": target, "entity_type": kind,
                    "target_is_source_domain": target in source_datasets,
                    "overlap_mentions": matches, "target_gold_mentions": len(names),
                    "overlap": matches / len(names),
                    "reference_f1": baseline[kind], "target_f1": scores["f1"],
                    "f1_loss_pp": (baseline[kind] - scores["f1"]) * 100,
                })
        by_source[source] = describe_relationship([p for p in source_points if not p["target_is_source_domain"]])
        points.extend(source_points)

    external = [point for point in points if not point["target_is_source_domain"]]
    report = {
        "matrix": MATRIX.relative_to(PROJECT).as_posix(),
        "overlap_definition": "Fraction of target gold mentions whose full case-folded name and mapped type occur in source training.",
        "loss_definition": "Same-source test F1 minus target F1, per type. For merged training, reference F1 is the mean of its two source-domain test F1s.",
        "regression_scope": "Target domains outside each training source; descriptive only, as source/target/type cells share data.",
        "all_non_source_targets": describe_relationship(external),
        "by_source_non_source_targets": by_source,
        "points": points,
    }
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        staged = Path(staging) / RESULTS.name
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(RESULTS)
    print(f"{len(external)} non-source cells: {report['all_non_source_targets']}")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
