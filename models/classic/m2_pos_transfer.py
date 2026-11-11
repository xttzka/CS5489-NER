"""Train one CoNLL-2003 M2 model and evaluate five complete test splits."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import read_tags_by_id
from eval.score_transfer import score_transfer
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_crf import train_model, write_predictions
from models.classic.m2_pos_validation import C1, C2, POS_MODEL, POS_ROOT, load_pos_rows, verify_alignment


SOURCE = "conll2003"
TARGETS = (SOURCE, "conllpp", "wikigold", "btc", "wnut17")
RESULTS = PROJECT_DIR / "results" / "m2_pos_transfer.json"


def main():
    train_path = POS_ROOT / SOURCE / "train.jsonl"
    training = load_pos_rows(train_path, "train", dataset=SOURCE)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    targets = {}
    pending_predictions = {}
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        work_dir = Path(staging)
        classifier = train_model(training, work_dir / "model.crfsuite", c1=C1, c2=C2)
        try:
            for dataset in TARGETS:
                test_path = POS_ROOT / dataset / "test.jsonl"
                gold_path = PROJECT_DIR / "data" / "processed" / dataset / test_path.name
                test_rows = load_pos_rows(test_path, "test", dataset=dataset)
                verify_alignment(gold_path, test_rows, "test")

                staged = work_dir / f"{dataset}.jsonl"
                repairs, sentences_with_repairs = write_predictions(staged, test_rows, classifier)
                scores = score_transfer(
                    read_tags_by_id(gold_path), read_tags_by_id(staged),
                    target_dataset=dataset, prediction_schema=SOURCE,
                )
                output = PROJECT_DIR / "preds" / dataset / "m2_pos_transfer" / "0.jsonl"
                pending_predictions[staged] = output
                targets[dataset] = {
                    "input": test_path.relative_to(PROJECT_DIR).as_posix(),
                    "gold": gold_path.relative_to(PROJECT_DIR).as_posix(),
                    "predictions": output.relative_to(PROJECT_DIR).as_posix(),
                    "scores": scores,
                    "raw_bio_error_tokens": repairs,
                    "raw_bio_error_sentences": sentences_with_repairs,
                }
                print(f"{dataset}: {scores['scored_records']:,} sentences, F1={scores['f1']:.4%}", flush=True)
        finally:
            classifier.tagger_.close()

        reference = targets[SOURCE]["scores"]
        for target in targets.values():
            target["f1_delta_vs_conll2003_pp"] = (target["scores"]["f1"] - reference["f1"]) * 100
        report = {
            "method": "M2 CRF",
            "source_dataset": SOURCE,
            "parameters": {"c1": C1, "c2": C2},
            "features": {"pos_model": POS_MODEL, "use_chunk_tags": False},
            "training_input": train_path.relative_to(PROJECT_DIR).as_posix(),
            "training_records": len(training),
            "evaluation": "full test splits",
            "entity_types": list(reference["per_type"]),
            "targets": targets,
        }
        staged_report = work_dir / "results.json"
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for staged, output in pending_predictions.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(output)
        staged_report.replace(RESULTS)

    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
