"""Compare CoNLL-2003, WNUT-17, and merged training in one shared label space."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import read_tags_by_id
from eval.score_transfer import MAPPING_PATH, project_tags, score_transfer
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_crf import train_model, write_predictions
from models.classic.m2_pos_transfer import RESULTS as CONLL_RESULTS
from models.classic.m2_pos_transfer import SOURCE, TARGETS
from models.classic.m2_pos_validation import C1, C2, POS_MODEL, POS_ROOT, load_pos_rows, verify_alignment


RESULTS = CONLL_RESULTS.with_name("m2_transfer_matrix.json")
GOLD_ROOT = PROJECT_DIR / "data" / "processed"
WNUT = "wnut17"
TRAINING_SOURCES = {SOURCE: (SOURCE,), WNUT: (WNUT,), "merged": (SOURCE, WNUT)}
PARAMETERS = {"c1": C1, "c2": C2}
FEATURES = {"pos_model": POS_MODEL, "use_chunk_tags": False}


def relative(path):
    return path.relative_to(PROJECT_DIR).as_posix()


def projected_training(dataset, label_map):
    path = POS_ROOT / dataset / "train.jsonl"
    rows = load_pos_rows(path, "train", dataset=dataset)
    verify_alignment(GOLD_ROOT / dataset / path.name, rows, "train")
    native = {row["sample_id"]: row["tags"] for row in rows}
    if not native or len(native) != len(rows):
        raise ValueError(f"{path}: expected a nonempty training set with unique sample IDs")
    projected = project_tags(native, label_map)
    for row in rows:
        row["tags"] = projected[row["sample_id"]]
    return rows


def evaluate_row(row_name, sources, train_rows, work_dir, predictions_to_publish):
    training = [row for dataset in sources for row in train_rows[dataset]]
    classifier = train_model(training, work_dir / f"{row_name}.crfsuite", **PARAMETERS)
    suffix = "conll2003_shared" if row_name == SOURCE else row_name
    targets = {}
    try:
        for dataset in TARGETS:
            input_path = POS_ROOT / dataset / "test.jsonl"
            gold_path = GOLD_ROOT / dataset / input_path.name
            rows = load_pos_rows(input_path, "test", dataset=dataset)
            verify_alignment(gold_path, rows, "test")

            staged = work_dir / f"{row_name}_{dataset}.jsonl"
            repairs, repaired_sentences = write_predictions(staged, rows, classifier)
            scores = score_transfer(
                read_tags_by_id(gold_path), read_tags_by_id(staged),
                target_dataset=dataset, prediction_schema=SOURCE,
            )
            path = PROJECT_DIR / "preds" / dataset / f"m2_pos_transfer_{suffix}" / "0.jsonl"
            predictions_to_publish[staged] = path
            targets[dataset] = {
                "input": relative(input_path),
                "gold": relative(gold_path),
                "predictions": relative(path),
                "scores": scores,
                "raw_bio_error_tokens": repairs,
                "raw_bio_error_sentences": repaired_sentences,
            }
            print(f"{row_name} -> {dataset}: {scores['scored_records']:,} sentences, F1={scores['f1']:.4%}", flush=True)
    finally:
        classifier.tagger_.close()

    return {
        "source_datasets": list(sources),
        "training_records": len(training),
        "training_records_by_dataset": {dataset: len(train_rows[dataset]) for dataset in sources},
        "training_inputs": [relative(POS_ROOT / dataset / "train.jsonl") for dataset in sources],
        "training_label_policy": "Project each training corpus to the shared three types before concatenation.",
        "targets": targets,
    }


def publish_results(report, work_dir, pending_predictions):
    staged_report = work_dir / "results.json"
    staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for staged, path in pending_predictions.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        staged.replace(path)
    staged_report.replace(RESULTS)


def main():
    settings = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["transfer_three_types"]
    original = json.loads(CONLL_RESULTS.read_text(encoding="utf-8"))
    if (
        original["source_dataset"] != SOURCE or original["parameters"] != PARAMETERS
        or original["features"] != FEATURES or original["entity_types"] != settings["types"]
        or set(original["targets"]) != set(TARGETS)
    ):
        raise ValueError(f"{CONLL_RESULTS}: incompatible CoNLL-2003 reference")

    train_rows = {
        dataset: projected_training(dataset, settings["maps"][dataset]) for dataset in (SOURCE, WNUT)
    }
    report = {
        "method": "M2 CRF",
        "parameters": PARAMETERS,
        "parameter_policy": "Reuse the CoNLL-2003 training-fold selection for all rows; no target-test tuning.",
        "features": FEATURES,
        "entity_types": settings["types"],
        "evaluation": "full test splits",
        "label_mapping": relative(MAPPING_PATH),
        "native_label_training_reference": relative(CONLL_RESULTS),
        "rows": {},
    }
    pending_predictions = {}
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        work_dir = Path(staging)
        for row_name, sources in TRAINING_SOURCES.items():
            report["rows"][row_name] = evaluate_row(row_name, sources, train_rows, work_dir, pending_predictions)
        publish_results(report, work_dir, pending_predictions)

    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
