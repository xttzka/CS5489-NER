"""Score one M2 model on the full original and three perturbed CoNLL-2003 tests."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from data.make_perturbation_sets import OUTPUT as PERTURBATIONS, SOURCE
from eval.score_ner import read_tags_by_id, score_ner
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_crf import train_model, write_predictions
from models.classic.m2_pos_validation import C1, C2, POS_DATA, POS_MODEL, PRED_ROOT, load_pos_rows


MANIFEST = PERTURBATIONS / "manifest_v1.json"
RESULTS = PROJECT_DIR / "results" / "m2_pos_perturbations.json"
INPUTS = {
    "original": SOURCE,
    "lowercase": PERTURBATIONS / "lowercase_v1.jsonl",
    "spelling_noise": PERTURBATIONS / "spelling_noise_v1.jsonl",
    "no_punctuation": PERTURBATIONS / "no_punctuation_v1.jsonl",
}


def load_test_variants():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    original_ids = None
    tests = {}
    for condition, path in INPUTS.items():
        expected_hash = manifest["source_sha256"] if condition == "original" else manifest["sha256"][path.name]
        if sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"{path}: file differs from the v1 perturbation manifest")
        rows = load_pos_rows(path, "test")
        ids = [row["sample_id"] for row in rows]
        if original_ids is None:
            original_ids = ids
        if ids != original_ids or len(ids) != manifest["sentences_per_variant"] or len(set(ids)) != len(ids):
            raise ValueError(f"{path}: expected the same complete test IDs as the original")
        tests[condition] = rows
    return tests


def main():
    tests = load_test_variants()
    train_path = POS_DATA / "train.jsonl"
    training = load_pos_rows(train_path, "train")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    conditions, pending = {}, {}
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        work_dir = Path(staging)
        classifier = train_model(training, work_dir / "model.crfsuite", c1=C1, c2=C2)
        try:
            for condition, rows in tests.items():
                staged = work_dir / f"{condition}.jsonl"
                repaired_tokens, repaired_sentences = write_predictions(staged, rows, classifier)
                # Deletion shifts offsets, so each answer is scored against its own variant's BIO.
                gold = {row["sample_id"]: row["tags"] for row in rows}
                scores = score_ner(gold, read_tags_by_id(staged))
                output = PRED_ROOT / f"m2_pos_perturbations_{condition}" / "0.jsonl"
                pending[staged] = output
                conditions[condition] = {
                    "input_and_gold": INPUTS[condition].relative_to(PROJECT_DIR).as_posix(),
                    "predictions": output.relative_to(PROJECT_DIR).as_posix(),
                    "scores": scores,
                    "raw_bio_error_tokens": repaired_tokens,
                    "raw_bio_error_sentences": repaired_sentences,
                }
                print(f"{condition}: F1={scores['f1']:.4%}, recall={scores['recall']:.4%}", flush=True)
        finally:
            classifier.tagger_.close()

        original = conditions["original"]["scores"]
        original_gold_entities = original["tp"] + original["fn"]
        for result in conditions.values():
            scores = result["scores"]
            if scores["tp"] + scores["fn"] != original_gold_entities:
                raise ValueError("perturbation changed the number of gold entities")
            result["delta_vs_original_pp"] = {
                metric: (scores[metric] - original[metric]) * 100 for metric in ("f1", "recall")
            }
        report = {
            "method": "M2 CRF",
            "parameters": {"c1": C1, "c2": C2},
            "features": {"pos_model": POS_MODEL, "pos_frozen_from_original": True, "use_chunk_tags": False},
            "training_input": train_path.relative_to(PROJECT_DIR).as_posix(),
            "training_records": len(training),
            "perturbation_manifest": MANIFEST.relative_to(PROJECT_DIR).as_posix(),
            "test_records_per_condition": original["scored_records"],
            "gold_entities_per_condition": original_gold_entities,
            "entity_types": ("PER", "LOC", "ORG", "MISC"),
            "conditions": conditions,
        }
        staged_report = work_dir / "results.json"
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for staged, output in pending.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(output)
        staged_report.replace(RESULTS)

    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
