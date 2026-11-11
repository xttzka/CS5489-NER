"""Measure how the same POS-only M2 responds to lowercased validation text."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import read_tags_by_id
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_crf import train_model, write_predictions
from models.classic.m2_pos_validation import (
    C1, C2, GOLD, POS_DATA, POS_MODEL, PRED_ROOT, load_pos_rows, score_validation, verify_alignment,
)


RESULTS = PROJECT_DIR / "results" / "m2_pos_casing.json"


def lowercase_rows(rows):
    # Keep the original token positions, POS tags, gold tags, and sample IDs.
    return [{**row, "tokens": [token.lower() for token in row["tokens"]]} for row in rows]


def metric_deltas(original, lowercase):
    return {metric: (lowercase[metric] - original[metric]) * 100 for metric in ("f1", "recall")}


def main():
    train_path = POS_DATA / "train.jsonl"
    validation_path = POS_DATA / "validation.jsonl"
    training = load_pos_rows(train_path, "train")
    original = load_pos_rows(validation_path, "validation")
    verify_alignment(GOLD, original, "validation")
    lowercase = lowercase_rows(original)
    gold = read_tags_by_id(GOLD)

    changed_sentences = changed_tokens = 0
    for before, after in zip(original, lowercase, strict=True):
        changed_sentences += before["tokens"] != after["tokens"]
        changed_tokens += sum(a != b for a, b in zip(before["tokens"], after["tokens"], strict=True))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    conditions = {}
    pending_predictions = {}
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        work_dir = Path(staging)
        classifier = train_model(training, work_dir / "model.crfsuite", c1=C1, c2=C2)
        try:
            for condition, rows in (("original", original), ("lowercase", lowercase)):
                staged = work_dir / f"{condition}.jsonl"
                repaired_tokens, repaired_sentences = write_predictions(staged, rows, classifier)
                scores = score_validation(gold, staged)
                output = PRED_ROOT / f"m2_pos_casing_{condition}" / "0.jsonl"
                pending_predictions[staged] = output
                conditions[condition] = {
                    "predictions": output.relative_to(PROJECT_DIR).as_posix(),
                    "scores": scores,
                    "raw_bio_error_tokens": repaired_tokens,
                    "raw_bio_error_sentences": repaired_sentences,
                }
                print(f"{condition}: F1={scores['f1']:.4%}, recall={scores['recall']:.4%}", flush=True)
        finally:
            classifier.tagger_.close()

        before, after = (conditions[name]["scores"] for name in ("original", "lowercase"))
        report = {
            "method": "M2 CRF",
            "parameters": {"c1": C1, "c2": C2},
            "features": {"pos_model": POS_MODEL, "use_chunk_tags": False},
            "training_input": train_path.relative_to(PROJECT_DIR).as_posix(),
            "training_records": len(training),
            "validation_input": validation_path.relative_to(PROJECT_DIR).as_posix(),
            "gold": GOLD.relative_to(PROJECT_DIR).as_posix(),
            "entity_types": list(before["per_type"]),
            "transformation": "Lowercase each validation token in place in its sequence; reuse original POS and gold.",
            "changed_sentences": changed_sentences,
            "changed_tokens": changed_tokens,
            "conditions": conditions,
            "lowercase_minus_original_pp": {
                "overall": metric_deltas(before, after),
                "per_type": {
                    kind: metric_deltas(before["per_type"][kind], after["per_type"][kind])
                    for kind in before["per_type"]
                },
            },
        }
        staged_report = work_dir / "results.json"
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for staged, output in pending_predictions.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(output)
        staged_report.replace(RESULTS)

    print(f"Changed {changed_tokens:,} tokens in {changed_sentences:,} validation sentences.")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
