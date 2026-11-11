"""Run the M2 validation trial with shared spaCy POS tags and no chunk features."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import read_tags_by_id
from eval.score_transfer import score_transfer
from models.classic.m0_dictionary import DATA_DIR, PROJECT_DIR, read_sentences
from models.classic.m2_crf import train_model, write_predictions


POS_ROOT = PROJECT_DIR / "data" / "processed_pos"
POS_DATA = POS_ROOT / "conll2003"
POS_MODEL = "en_core_web_sm 3.8.0"
GOLD = DATA_DIR / "validation.jsonl"
PRED_ROOT = PROJECT_DIR / "preds" / "conll2003"
OLD_PREDICTIONS = PRED_ROOT / "m2_cv_grid" / "0.jsonl"
PREDICTIONS = PRED_ROOT / "m2_pos" / "0.jsonl"
RESULTS = PROJECT_DIR / "results" / "m2_pos_validation.json"
C1 = 0.0001
C2 = 0.1


def load_pos_rows(path, split, *, dataset="conll2003"):
    rows = list(read_sentences(path, split))
    for row in rows:
        if row["dataset"] != dataset or row.get("pos_tags") is None:
            raise ValueError(f"{row['sample_id']}: expected {dataset} with spaCy POS tags")
        # The shared feature extractor skips the chunk feature when its input is None.
        row["chunk_tags"] = None
    return rows


def verify_alignment(original_path, rows, split):
    for original, prepared in zip(read_sentences(original_path, split), rows, strict=True):
        if any(original[field] != prepared[field] for field in ("sample_id", "tokens", "tags")):
            raise ValueError(f"{prepared['sample_id']}: POS version differs from {original_path}")


def score_validation(gold, prediction_path):
    return score_transfer(
        gold, read_tags_by_id(prediction_path),
        target_dataset="conll2003", prediction_schema="conll2003",
    )


def main():
    train_path = POS_DATA / "train.jsonl"
    validation_path = POS_DATA / "validation.jsonl"
    training = load_pos_rows(train_path, "train")
    validation = load_pos_rows(validation_path, "validation")
    verify_alignment(GOLD, validation, "validation")

    gold = read_tags_by_id(GOLD)
    previous = score_validation(gold, OLD_PREDICTIONS)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        directory = Path(staging)
        classifier = train_model(training, directory / "model.crfsuite", c1=C1, c2=C2)
        predictions = directory / "predictions.jsonl"
        try:
            repaired_tokens, repaired_sentences = write_predictions(predictions, validation, classifier)
        finally:
            classifier.tagger_.close()

        current = score_validation(gold, predictions)
        delta = current["f1"] - previous["f1"]
        report = {
            "method": "M2 CRF",
            "parameters": {"c1": C1, "c2": C2},
            "features": {"pos_model": POS_MODEL, "use_chunk_tags": False},
            "training_input": train_path.relative_to(PROJECT_DIR).as_posix(),
            "training_records": len(training),
            "validation_input": validation_path.relative_to(PROJECT_DIR).as_posix(),
            "gold": GOLD.relative_to(PROJECT_DIR).as_posix(),
            "entity_types": list(current["per_type"]),
            "old_m2": {
                "predictions": OLD_PREDICTIONS.relative_to(PROJECT_DIR).as_posix(),
                "scores": previous,
            },
            "uniform_pos_m2": {
                "predictions": PREDICTIONS.relative_to(PROJECT_DIR).as_posix(),
                "scores": current,
                "raw_bio_error_tokens": repaired_tokens,
                "raw_bio_error_sentences": repaired_sentences,
            },
            "f1_delta_percentage_points": delta * 100,
        }
        results = directory / "results.json"
        results.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        predictions.replace(PREDICTIONS)
        results.replace(RESULTS)

    print(f"Three-type validation F1: old={previous['f1']:.4%}, new={current['f1']:.4%} ({delta * 100:+.4f} pp)")
    print(f"Predictions: {PREDICTIONS}")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
