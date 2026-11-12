"""Train M0/M1 once and compare M0–M2 on both complete news test splits."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_familiarity import FAMILIARITY, REFERENCE, news_settings, score_full_test, training_names
from eval.score_m2_familiarity import TRANSFER_REPORT
from models.classic import m0_dictionary, m1_logistic
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_pos_validation import POS_MODEL
from models.classic.run_baselines import load_rows, m2_selection, read_m1_selection, relative


SOURCE = "conll2003"
RESULTS = PROJECT_DIR / "results" / "m0_m2_temporal.json"
GROUPS = ("overall", "seen", "unseen")


def main():
    settings, names = news_settings(), training_names()
    label_map = {kind: kind for kind in settings["types"]}
    train_path, rows = load_rows(SOURCE, "train", shared_pos=True)
    m2 = json.loads(TRANSFER_REPORT.read_text(encoding="utf-8"))
    features = {"pos_model": POS_MODEL, "use_chunk_tags": False}
    m2_config = m2_selection(SOURCE, train_path, {**features, "context_window": 1})
    if (m2["source_dataset"] != SOURCE or m2["training_input"] != relative(train_path)
            or m2["training_records"] != len(rows) or m2["features"] != features
            or m2["parameters"] != m2_config["parameters"]):
        raise ValueError("M2 report does not use the same source data and shared POS configuration")

    training = list(rows.values())
    dictionary, ambiguous = m0_dictionary.build_dictionary(training)
    m1_config = read_m1_selection(SOURCE, train_path)
    vectorizer, classifier = m1_logistic.train_model(training, c=m1_config["c"])
    report = {
        "source_dataset": SOURCE,
        "training_input": relative(train_path),
        "training_records": len(training),
        "entity_types": settings["types"],
        "evaluation": "Full test splits; exact match on sample ID, token boundaries and entity type.",
        "training_name_reference": relative(REFERENCE),
        "familiarity": FAMILIARITY,
        "features": {**features, "context_window": 1},
        "models": {
            "m0": {"dictionary_entries": len(dictionary), "ambiguous_names_excluded": ambiguous},
            "m1": {**m1_config, "iterations": int(classifier.n_iter_[0])},
            "m2": {"parameters": m2["parameters"], "reused_report": relative(TRANSFER_REPORT)},
        },
        "datasets": {},
    }
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        stage = Path(staging)
        pending = {}
        for dataset in settings["datasets"]:
            test_path, rows = load_rows(dataset, "test", shared_pos=True)
            test = list(rows.values())
            m2_target = m2["targets"][dataset]
            if m2_target["input"] != relative(test_path):
                raise ValueError(f"{dataset}: M2 test input differs from the current shared POS input")
            answers = {model: stage / f"{dataset}_{model}.jsonl" for model in ("m0", "m1")}
            m0_dictionary.write_predictions(answers["m0"], test, dictionary)
            m1_bio = m1_logistic.write_predictions(answers["m1"], test, vectorizer, classifier)
            for model, path in answers.items():
                pending[path] = PROJECT_DIR / "preds" / dataset / f"{model}_temporal" / "0.jsonl"
            answers["m2"] = PROJECT_DIR / m2_target["predictions"]

            scored = {}
            for model, path in answers.items():
                scores = score_full_test(dataset, path, names, label_map)
                scores["predictions"] = relative(pending.get(path, path))
                scored[model] = scores
            scored["m1"]["raw_bio_repairs"] = {"tokens": m1_bio[0], "sentences": m1_bio[1]}
            report["datasets"][dataset] = scored

        current, later = (report["datasets"][dataset] for dataset in (SOURCE, "conllpp"))
        report["conllpp_minus_conll2003_f1_pp"] = {
            model: {key: 100 * (later[model][key]["f1"] - current[model][key]["f1"]) for key in GROUPS}
            for model in current
        }

        staged_report = stage / RESULTS.name
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for path, output in pending.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            path.replace(output)
        staged_report.replace(RESULTS)

    for dataset, scores in report["datasets"].items():
        for model, row in scores.items():
            summary = ", ".join(f"{key}={row[key]['f1']:.2%}" for key in GROUPS)
            print(f"{dataset} {model.upper()} ({row['test_records']:,} sentences): {summary}")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
