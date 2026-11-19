"""Reproduce A's fixed CoNLL M0/M1/M2 on validation for B's selection.

Only the CoNLL train and validation splits are read. Hyperparameters and the
uniform spaCy POS feature definition come from A's already completed report;
this script does not search parameters or write A's files. Use a new directory
under output/ for each invocation. Incomplete directories are preserved.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from eval.score_ner import read_tags_by_id, score_ner


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "conll2003"
POS = ROOT / "data" / "processed_pos" / "conll2003"
REPORT = ROOT / "results" / "m0_m2_baselines.json"
CV_REPORT = ROOT / "results" / "m2_cv_grid.json"
M2_REPORT = ROOT / "results" / "m2_pos_validation.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as target:
        json.dump(value, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")


def selection_config():
    """Verify the exact A selection and data hashes, without reading test data."""
    from models.classic import m1_logistic
    from models.classic.run_baselines import m2_selection, read_m1_selection

    train_path, val_path = POS / "train.jsonl", POS / "validation.jsonl"
    m1 = read_m1_selection("conll2003", train_path)
    selection = m1["selection"]
    if selection["split"] != "conll2003:validation" or selection["validation_sha256"] != sha256(val_path):
        raise ValueError("A's M1 validation inputs do not match the POS validation file")
    if m1["c"] != 10.0 or (m1["solver"], m1["tol"], m1["max_iterations"]) != (
        m1_logistic.SOLVER, m1_logistic.TOL, m1_logistic.MAX_ITER
    ):
        raise ValueError("A's M1 fixed hyperparameters have changed")
    features = {"pos_model": "en_core_web_sm 3.8.0", "use_chunk_tags": False, "context_window": 1}
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report["groups"]["conll2003"]["m1_m2_features"] != features:
        raise ValueError("A's final feature definition differs")
    m2 = m2_selection("conll2003", train_path, features)
    if m2["parameters"] != {"c1": 0.0001, "c2": 0.1}:
        raise ValueError("A's M2 fixed hyperparameters have changed")
    return m1, m2, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new directory under this project's output/ directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output_root = (ROOT / "output").resolve()
    if output == output_root or not output.is_relative_to(output_root):
        parser.error("--output-dir must be a new subdirectory of output/")
    if output.exists():
        parser.error(f"output exists; refusing to overwrite: {output}")

    from models.classic import m0_dictionary, m1_logistic, m2_crf
    from models.classic.m2_pos_validation import load_pos_rows, verify_alignment
    from eval.score_transfer import score_transfer

    m1, m2, features = selection_config()
    train = load_pos_rows(POS / "train.jsonl", "train")
    validation = load_pos_rows(POS / "validation.jsonl", "validation")
    verify_alignment(DATA / "train.jsonl", train, "train")
    verify_alignment(DATA / "validation.jsonl", validation, "validation")
    if len(train) != 14041 or len(validation) != 3250:
        raise ValueError("unexpected CoNLL train or validation record count")
    train_ids = {row["sample_id"] for row in train}
    val_ids = {row["sample_id"] for row in validation}
    if len(train_ids) != len(train) or len(val_ids) != len(validation) or train_ids & val_ids:
        raise ValueError("duplicate or overlapping train and validation IDs")
    gold = read_tags_by_id(DATA / "validation.jsonl")

    output.mkdir(parents=True, exist_ok=False)
    evidence = {
        "dataset": "conll2003", "split": "validation", "scope": "full",
        "label_space": "LOC+MISC+ORG+PER",
        "classic_configuration_note": (
            "A's fixed CoNLL baselines: case-sensitive longest-match dictionary; "
            "M1 C=10 lbfgs; M2 c1=0.0001 c2=0.1 lbfgs; "
            "M1/M2 use en_core_web_sm 3.8.0 POS, no chunk and context window 1. "
            "M1 C was selected by A on this validation split; M2 by A's train-only CV."
        ),
        "features": features,
        "m1_config": {key: m1[key] for key in ("c", "solver", "tol", "max_iterations")},
        "m2_config": m2,
        "inputs_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                          (DATA / "train.jsonl", DATA / "validation.jsonl",
                           POS / "train.jsonl", POS / "validation.jsonl", REPORT, CV_REPORT, M2_REPORT)},
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                        (Path(__file__), ROOT / "models/classic/features.py",
                         ROOT / "models/classic/m0_dictionary.py", ROOT / "models/classic/m1_logistic.py",
                         ROOT / "models/classic/m2_crf.py", ROOT / "models/classic/m2_pos_validation.py")},
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "scikit-learn", "sklearn-crfsuite", "python-crfsuite", "threadpoolctl")},
        "python": platform.python_version(), "predictions": [], "scores": {},
    }
    write_json(output / "run_config.json", evidence)
    started = time.monotonic()

    print("M0: building dictionary", flush=True)
    dictionary, ambiguous = m0_dictionary.build_dictionary(train)
    if len(dictionary) != 7950 or ambiguous != 132:
        raise ValueError("dictionary counts differ from A's recorded training data")
    m0_path = output / "m0_validation.jsonl"
    m0_dictionary.write_predictions(m0_path, validation, dictionary)

    print("M1: training fixed C=10", flush=True)
    vectorizer, model = m1_logistic.train_model(train, c=m1["c"])
    m1_path = output / "m1_validation.jsonl"
    repairs1 = m1_logistic.write_predictions(m1_path, validation, vectorizer, model)
    del vectorizer, model

    print("M2: training fixed c1=0.0001, c2=0.1", flush=True)
    m2_path = output / "m2_validation.jsonl"
    crf = m2_crf.train_model(train, output / "m2_model.crfsuite", **m2["parameters"])
    try:
        repairs2 = m2_crf.write_predictions(m2_path, validation, crf)
    finally:
        crf.tagger_.close()

    for method, path in (("M0", m0_path), ("M1", m1_path), ("M2", m2_path)):
        predictions = read_tags_by_id(path)
        evidence["predictions"].append({"method": method, "seed": 0, "path": str(path), "sha256": sha256(path)})
        evidence["scores"][method] = score_ner(gold, predictions)
    evidence["m1_bio_repairs"] = {"tokens": repairs1[0], "sentences": repairs1[1]}
    evidence["m2_bio_repairs"] = {"tokens": repairs2[0], "sentences": repairs2[1]}
    evidence["m2_shared_three_validation"] = score_transfer(
        gold, read_tags_by_id(m2_path), target_dataset="conll2003", prediction_schema="conll2003")
    m1_reference = next(item for item in m1["selection"]["candidates"] if item["c"] == m1["c"])
    m2_reference = json.loads(M2_REPORT.read_text(encoding="utf-8"))["uniform_pos_m2"]["scores"]
    evidence["reference_comparison"] = {
        "m1_four_type_f1_a": m1_reference["f1"],
        "m1_four_type_f1_reproduced": evidence["scores"]["M1"]["f1"],
        "m1_difference": evidence["scores"]["M1"]["f1"] - m1_reference["f1"],
        "m2_three_type_f1_a": m2_reference["f1"],
        "m2_three_type_f1_reproduced": evidence["m2_shared_three_validation"]["f1"],
        "m2_difference": evidence["m2_shared_three_validation"]["f1"] - m2_reference["f1"],
        "note": "Reference scores are checks, not inputs to model selection; software versions may affect reproduction.",
    }
    evidence["elapsed_seconds"] = time.monotonic() - started
    evidence["status"] = "complete"
    write_json(output / "classic_evidence.json", evidence)
    for method, scores in evidence["scores"].items():
        print(f"{method} full four-type validation F1: {scores['f1']:.6f}", flush=True)
    print("Difference from A's recorded validation F1: "
          f"M1={evidence['reference_comparison']['m1_difference']:+.8f}, "
          f"M2 (three-type)={evidence['reference_comparison']['m2_difference']:+.8f}", flush=True)
    print(f"Evidence: {output / 'classic_evidence.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
