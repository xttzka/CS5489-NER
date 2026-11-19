"""Freeze A's classic settings, then recreate token-aligned held-out predictions.

The prepare command never opens a test split. Prediction requires both this
script's immutable preparation manifest and B's frozen evaluation manifest.
It fits A's fixed M0/M1/M2 using only the chosen source training split; no test
scores are calculated here. Existing artifacts and incomplete runs are kept.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ("conll2003", "wnut17", "mit_restaurant")
FEATURES = {"pos_model": "en_core_web_sm 3.8.0", "use_chunk_tags": False, "context_window": 1}
TARGETS = {
    "conll2003": ["conll2003", "conllpp", "wikigold", "btc"],
    "wnut17": ["wnut17"],
    "mit_restaurant": ["mit_restaurant"],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def prepare_plan():
    """Check official fixed choices and training provenance without test data."""
    from models.classic.run_baselines import m2_selection, read_m1_selection

    report_path = ROOT / "results/m0_m2_baselines.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        raise ValueError("A's baseline report is incomplete")
    configs = {}
    for source in SOURCES:
        train = ROOT / "data/processed_pos" / source / "train.jsonl"
        validation = train.with_name("validation.jsonl")
        m1 = read_m1_selection(source, train)
        if m1["selection"]["validation_sha256"] != sha256(validation):
            raise ValueError(f"{source}: POS validation data differs from A's report")
        if report["groups"][source]["m1_m2_features"] != FEATURES:
            raise ValueError(f"{source}: final feature configuration differs from A's report")
        m2 = m2_selection(source, train, FEATURES)
        if m2["parameters"] != report["groups"][source]["m2"]["parameters"]:
            raise ValueError(f"{source}: selected M2 parameters differ from official report")
        files = [train, validation, ROOT / "data/processed" / source / "train.jsonl",
                 ROOT / "data/processed" / source / "validation.jsonl"]
        configs[source] = {
            "m1": {key: m1[key] for key in ("c", "solver", "tol", "max_iterations")},
            "m2": m2, "features": FEATURES,
            "inputs_sha256": {relative(file): sha256(file) for file in files},
        }
    code = [Path(__file__), ROOT / "models/classic/features.py", ROOT / "models/classic/m0_dictionary.py",
            ROOT / "models/classic/m1_logistic.py", ROOT / "models/classic/m2_crf.py",
            ROOT / "models/classic/m2_pos_validation.py", ROOT / "models/classic/run_baselines.py",
            ROOT / "eval/score_ner.py", ROOT / "eval/score_transfer.py", ROOT / "data/ner_label_mapping.json",
            report_path, ROOT / "results/m2_cv_grid.json", ROOT / "results/m2_native_cv.json"]
    plan = {
        "version": 1, "state": "classic_fixed", "sources": configs,
        "targets": TARGETS, "artifacts_sha256": {relative(path): sha256(path) for path in code},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scikit-learn", "sklearn-crfsuite", "python-crfsuite", "threadpoolctl")},
        "python": platform.python_version(),
        "policy": "Training and tuning choices from A's existing report; no test metrics or tuning.",
    }
    plan["manifest_sha256"] = digest_json(plan)
    return plan


def read_manifest(path, state):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = data.pop("manifest_sha256", None)
    if data.get("version") != 1 or data.get("state") != state or digest != digest_json(data):
        raise ValueError(f"invalid or modified {state} manifest: {path}")
    data["manifest_sha256"] = digest
    return data


def token_rows(dataset, ids=None):
    """Yield aligned original tokens/POS only, after formal freeze."""
    from models.classic.m2_pos_validation import load_pos_rows
    from models.classic.m0_dictionary import read_sentences

    pos_path = ROOT / "data/processed_pos" / dataset / "test.jsonl"
    original = ROOT / "data/processed" / dataset / "test.jsonl"
    prepared = load_pos_rows(pos_path, "test", dataset=dataset)
    rows = []
    for original_row, row in zip(read_sentences(original, "test"), prepared, strict=True):
        if any(original_row[key] != row[key] for key in ("sample_id", "tokens")):
            raise ValueError(f"{dataset}: POS text/ID alignment differs for {row['sample_id']}")
        rows.append({"sample_id": row["sample_id"], "tokens": row["tokens"],
                     "pos_tags": row["pos_tags"], "chunk_tags": None})
    by_id = {row["sample_id"]: row for row in rows}
    if len(by_id) != len(rows) or not rows:
        raise ValueError(f"{dataset}: duplicate IDs or empty input")
    if ids is not None:
        if len(ids) != 300 or len(set(ids)) != 300 or set(ids) - by_id.keys():
            raise ValueError(f"{dataset}: invalid fixed 300 IDs")
        rows = [by_id[sample_id] for sample_id in ids]
    return rows


def filter_predictions(full_path, subset_path, ids):
    from eval.score_ner import read_tags_by_id
    predictions = read_tags_by_id(full_path)
    if set(ids) - predictions.keys():
        raise ValueError("full predictions omit fixed-subset IDs")
    with subset_path.open("x", encoding="utf-8") as target:
        for sample_id in ids:
            target.write(json.dumps({"sample_id": sample_id, "tags": predictions[sample_id]},
                                    ensure_ascii=False) + "\n")


def predict(plan, frozen, source, output):
    import joblib
    from eval.score_ner import read_subset_ids, read_tags_by_id, score_ner
    from models.classic import m0_dictionary, m1_logistic, m2_crf
    from models.classic.m2_pos_validation import load_pos_rows, verify_alignment

    # Recompute the plan before accessing any test split. This checks the
    # fixed parameters, package versions, source data, and inference code.
    if plan != prepare_plan():
        raise ValueError("classic plan or its inputs/code/versions changed; re-audit before proceeding")
    if frozen.get("partial") or not frozen.get("selection") or not frozen.get("tasks"):
        raise ValueError("incomplete B frozen evaluation manifest")
    if output.exists():
        raise FileExistsError(f"output exists; refusing overwrite: {output}")
    if not output.is_relative_to((ROOT / "output").resolve()):
        raise ValueError("output must be under project output/")
    train = load_pos_rows(ROOT / "data/processed_pos" / source / "train.jsonl", "train", dataset=source)
    dev = load_pos_rows(ROOT / "data/processed_pos" / source / "validation.jsonl", "validation", dataset=source)
    verify_alignment(ROOT / "data/processed" / source / "train.jsonl", train, "train")
    verify_alignment(ROOT / "data/processed" / source / "validation.jsonl", dev, "validation")
    train_ids = {row["sample_id"] for row in train}
    dev_ids = {row["sample_id"] for row in dev}
    if len(train_ids) != len(train) or len(dev_ids) != len(dev) or train_ids & dev_ids:
        raise ValueError("duplicate/overlapping source train and validation IDs")

    output.mkdir(parents=True, exist_ok=False)
    cfg = plan["sources"][source]
    metadata = {"source": source, "classic_manifest_sha256": plan["manifest_sha256"],
                "frozen_manifest_sha256": frozen["manifest_sha256"], "config": cfg,
                "training_seconds": {}, "validation": {}, "predictions": [], "device": "cpu"}
    started = time.perf_counter()
    dictionary, ambiguous = m0_dictionary.build_dictionary(train)
    metadata["training_seconds"]["M0"] = time.perf_counter() - started
    metadata["dictionary"] = {"entries": len(dictionary), "ambiguous_names_excluded": ambiguous}
    started = time.perf_counter()
    vectorizer, logistic = m1_logistic.train_model(train, c=cfg["m1"]["c"])
    metadata["training_seconds"]["M1"] = time.perf_counter() - started
    started = time.perf_counter()
    crf = m2_crf.train_model(train, output / "m2_model.crfsuite", **cfg["m2"]["parameters"])
    metadata["training_seconds"]["M2"] = time.perf_counter() - started

    try:
        joblib.dump(dictionary, output / "m0_dictionary.joblib")
        joblib.dump({"vectorizer": vectorizer, "classifier": logistic}, output / "m1_model.joblib")
        model_hashes = {method: sha256(output / name) for method, name in (
            ("M0", "m0_dictionary.joblib"), ("M1", "m1_model.joblib"), ("M2", "m2_model.crfsuite"))}
        metadata["model_sha256"] = model_hashes
        # Dev comparison is an audit, not a model or hyperparameter selection.
        dev_gold = read_tags_by_id(ROOT / "data/processed" / source / "validation.jsonl")
        for method in ("M0", "M1", "M2"):
            path = output / f"{method.lower()}_validation.jsonl"
            if method == "M0":
                m0_dictionary.write_predictions(path, dev, dictionary)
            elif method == "M1":
                m1_logistic.write_predictions(path, dev, vectorizer, logistic)
            else:
                m2_crf.write_predictions(path, dev, crf)
            metadata["validation"][method] = score_ner(dev_gold, read_tags_by_id(path))
        write_json(output / "pretest_config.json", metadata)

        for target in TARGETS[source]:
            subset_path = ROOT / "data/subsets" / target / "test_300.txt"
            input_paths = (ROOT / "data/processed" / target / "test.jsonl",
                           ROOT / "data/processed_pos" / target / "test.jsonl", subset_path)
            inputs = {relative(path): sha256(path) for path in input_paths}
            ids = read_subset_ids(subset_path)
            is_domain = target == source
            rows = token_rows(target, None if is_domain else ids)
            if any(sha256(path) != inputs[relative(path)] for path in input_paths):
                raise ValueError(f"{target}: test input changed while being loaded")
            if {row["sample_id"] for row in rows} & (train_ids | dev_ids):
                raise ValueError(f"{target}: source train/validation IDs overlap test IDs")
            for method in ("M0", "M1", "M2"):
                scope = "full" if is_domain else "test_300"
                task_id = f"{target}__{method}__{scope}"
                task_dir = output / "tasks" / task_id
                task_dir.mkdir(parents=True, exist_ok=False)
                full = task_dir / "predictions.jsonl"
                started = time.perf_counter()
                if method == "M0":
                    m0_dictionary.write_predictions(full, rows, dictionary)
                    repairs = (0, 0)
                elif method == "M1":
                    repairs = m1_logistic.write_predictions(full, rows, vectorizer, logistic)
                else:
                    repairs = m2_crf.write_predictions(full, rows, crf)
                duration = time.perf_counter() - started
                record = {"task_id": task_id, "method": method, "train_source": source, "target": target,
                          "split": "test", "scope": scope, "path": str(full),
                          "prediction_sha256": sha256(full), "records": len(rows), "inference_seconds": duration,
                          "sentences_per_second": len(rows) / duration if duration else None,
                          "invalid_bio_tokens_repaired": repairs[0], "invalid_bio_sentences_repaired": repairs[1],
                          "inputs_sha256": inputs, "subset_sha256": inputs[relative(subset_path)],
                          "checkpoint_sha256": model_hashes[method],
                          "classic_manifest_sha256": plan["manifest_sha256"],
                          "frozen_manifest_sha256": frozen["manifest_sha256"], "train_config": cfg,
                          "versions": plan["versions"], "training_seconds": metadata["training_seconds"][method],
                          "prediction_schema": source,
                          "target_schema": "btc" if target == "btc" else None}
                write_json(task_dir / "metadata.json", record)
                metadata["predictions"].append(record)
                if is_domain:
                    subset_id = f"{target}__{method}__test_300"
                    subset_dir = output / "tasks" / subset_id
                    subset_dir.mkdir(parents=True, exist_ok=False)
                    subset = subset_dir / "predictions.jsonl"
                    if len(ids) != 300 or len(set(ids)) != 300:
                        raise ValueError(f"{target}: invalid fixed-300 list")
                    filter_predictions(full, subset, ids)
                    subset_record = {**record, "task_id": subset_id, "scope": "test_300", "path": str(subset),
                                     "prediction_sha256": sha256(subset), "records": 300,
                                     "inference_seconds": None, "sentences_per_second": None,
                                     "derived_from": str(full), "invalid_bio_tokens_repaired": None,
                                     "invalid_bio_sentences_repaired": None}
                    write_json(subset_dir / "metadata.json", subset_record)
                    metadata["predictions"].append(subset_record)
            print(f"Predictions ready: {source} -> {target}", flush=True)
    finally:
        crf.tagger_.close()
    metadata["status"] = "complete"
    write_json(output / "metadata.json", metadata)
    print(f"Done: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("prepare", help="Freeze A's configurations; reads no test split")
    pre.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("predict", help="Train one fixed source and predict after formal B freeze")
    run.add_argument("--classic-plan", type=Path, required=True)
    run.add_argument("--frozen", type=Path, required=True)
    run.add_argument("--source", choices=SOURCES, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            plan = prepare_plan()
            write_json(args.output, plan)
            print(f"Classic plan frozen: {args.output} ({plan['manifest_sha256']})")
        else:
            plan = read_manifest(args.classic_plan, "classic_fixed")
            frozen = read_manifest(args.frozen, "frozen")
            predict(plan, frozen, args.source, args.output_dir.resolve())
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
