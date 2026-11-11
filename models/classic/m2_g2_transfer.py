"""Compare explicit G2 features on three native tests and CoNLL-to-BTC."""

import json
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import read_tags_by_id, score_ner
from eval.score_transfer import project_tags, score_transfer
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_feature_ablation import PARAMETERS, features, fit_feature_model, predict_feature_rows
from models.classic.m2_pos_validation import POS_MODEL, POS_ROOT, load_pos_rows, verify_alignment


TRANSFER_REPORT = PROJECT_DIR / "results" / "m2_pos_transfer.json"
RESULTS = PROJECT_DIR / "results" / "m2_g2_transfer.json"
TASKS = {"conll2003": ("conll2003", "btc"), "wnut17": ("wnut17",), "mit_restaurant": ("mit_restaurant",)}


def score_target(source, target, rows, answers):
    gold = {row["sample_id"]: row["tags"] for row in rows}
    if source != target:
        return score_transfer(gold, answers, target_dataset=target, prediction_schema=source)
    types = sorted({tag[2:] for data in (gold, answers) for tags in data.values() for tag in tags if tag != "O"})
    scores = score_ner(gold, answers)
    scores["per_type"] = {}
    for kind in types:
        label_map = {tag: tag if tag == kind else None for tag in types}
        scores["per_type"][kind] = score_ner(project_tags(gold, label_map), project_tags(answers, label_map))
    return scores


def main():
    cached = json.loads(TRANSFER_REPORT.read_text(encoding="utf-8"))
    if (cached["parameters"] != PARAMETERS
            or cached["features"] != {"pos_model": POS_MODEL, "use_chunk_tags": False}):
        raise ValueError("the saved CoNLL baseline uses a different feature or parameter configuration")
    test = {}
    for target in {target for targets in TASKS.values() for target in targets}:
        path = POS_ROOT / target / "test.jsonl"
        rows = load_pos_rows(path, "test", dataset=target)
        verify_alignment(PROJECT_DIR / "data" / "processed" / target / path.name, rows, "test")
        test[target] = rows

    report = {
        "parameters": PARAMETERS,
        "pos_model": POS_MODEL,
        "use_chunk_tags": False,
        "comparison": "Keep all existing features except G2: capitalization, shape, digit and hyphen flags. G1 still includes the original word.",
        "domain_tuning": "Fixed parameters from CoNLL-2003 training CV for both variants and all sources.",
        "rows": {},
    }
    pending = {}
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        work_dir = Path(staging)
        for source, targets in TASKS.items():
            train_path = POS_ROOT / source / "train.jsonl"
            training = load_pos_rows(train_path, "train", dataset=source)
            outputs = {target: {} for target in targets}
            report["rows"][source] = {
                "training_input": train_path.relative_to(PROJECT_DIR).as_posix(),
                "training_records": len(training), "targets": outputs,
            }
            for condition in ("full", "without_G2"):
                reuse = source == "conll2003" and condition == "full"
                extract = partial(features, condition=condition)
                classifier = None
                try:
                    if not reuse:
                        print(f"Training {source} {condition} on {len(training):,} sentences", flush=True)
                        classifier = fit_feature_model(training, work_dir / f"{source}_{condition}.crfsuite", extract)
                    for target in targets:
                        if reuse:
                            path = PROJECT_DIR / cached["targets"][target]["predictions"]
                            answers = read_tags_by_id(path)
                            counts = {key: cached["targets"][target][key] for key in (
                                "raw_bio_error_tokens", "raw_bio_error_sentences",
                            )}
                        else:
                            answers, counts = predict_feature_rows(test[target], classifier, extract)
                            stem = f"m2_g2_{source}_{condition}"
                            staged = work_dir / f"{stem}_{target}.jsonl"
                            with staged.open("w", encoding="utf-8", newline="\n") as file:
                                for sample_id, tags in answers.items():
                                    file.write(json.dumps({"sample_id": sample_id, "tags": tags}) + "\n")
                            path = PROJECT_DIR / "preds" / target / stem / "0.jsonl"
                            pending[staged] = path
                        scores = score_target(source, target, test[target], answers)
                        outputs[target][condition] = {
                            "predictions": path.relative_to(PROJECT_DIR).as_posix(),
                            "reused": reuse, "scores": scores, **counts,
                        }
                        print(f"{source} -> {target} {condition}: F1={scores['f1']:.4%}", flush=True)
                finally:
                    if classifier is not None:
                        classifier.tagger_.close()
            for target in targets:
                row = outputs[target]
                row["without_G2_minus_full_f1_pp"] = (
                    row["without_G2"]["scores"]["f1"] - row["full"]["scores"]["f1"]
                ) * 100

        staged_report = work_dir / "results.json"
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for staged, output in pending.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(output)
        staged_report.replace(RESULTS)
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
