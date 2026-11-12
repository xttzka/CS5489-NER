"""Train M0–M2 and score the full in-domain tests and the six fixed subsets."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.baseline_statistics import MODELS, score_baselines
from eval.score_ner import read_subset_ids, read_tags_by_id
from eval.score_transfer import MAPPING_PATH, project_tags
from models.classic import m0_dictionary, m1_logistic, m2_crf
from models.classic.m0_dictionary import PROJECT_DIR, read_sentences
from models.classic.m2_pos_validation import POS_MODEL, POS_ROOT, load_pos_rows, verify_alignment


DATA = PROJECT_DIR / "data" / "processed"
SUBSETS = PROJECT_DIR / "data" / "subsets"
RESULTS = PROJECT_DIR / "results" / "m0_m2_baselines.json"
CO_NLL_CV = RESULTS.with_name("m2_cv_grid.json")
NATIVE_CV = RESULTS.with_name("m2_native_cv.json")
GROUPS = {
    "conll2003": ("conll2003", True, ("conll2003",)),
    "wnut17": ("wnut17", True, ("wnut17",)),
    "mit_restaurant": ("mit_restaurant", True, ("mit_restaurant",)),
    "transfer": ("conll2003", True, ("conllpp", "wikigold", "btc")),
}
SOURCES = ("conll2003", "wnut17", "mit_restaurant")


def relative(path):
    return path.relative_to(PROJECT_DIR).as_posix()


def load_rows(dataset, split, shared_pos):
    original = DATA / dataset / f"{split}.jsonl"
    if shared_pos:
        path = POS_ROOT / dataset / original.name
        rows = load_pos_rows(path, split, dataset=dataset)
        verify_alignment(original, rows, split)
    else:
        path = original
        rows = list(read_sentences(path, split))
    rows_by_id = {row["sample_id"]: row for row in rows}
    if len(rows_by_id) != len(rows) or not rows:
        raise ValueError(f"{path}: expected nonempty data with unique IDs")
    return path, rows_by_id


def m2_selection(group, train_path, features):
    if group == "transfer":
        group = "conll2003"

    path = CO_NLL_CV if group == "conll2003" else NATIVE_CV
    report = json.loads(path.read_text(encoding="utf-8"))
    if group == "conll2003":
        selection = report
        if selection["status"] != "complete" or selection["context_window"] != features["context_window"]:
            raise ValueError(f"{path}: expected a complete grid using the shared context window")
        train_hash = selection["fingerprint"]["train_sha256"]
        best = next(candidate for candidate in selection["candidates"] if all(
            candidate[key] == value for key, value in selection["best_parameters"].items()
        ))
        best_mean = best["mean_f1"]
        fold_strategy = "Five held-out folds grouped by doc_id."
        cv_training = DATA / "conll2003" / "train.jsonl"
        cv_features = {"pos_model": "original CoNLL-2003 POS", "use_chunk_tags": True, "context_window": 1}
    else:
        if report["features"] != features:
            raise ValueError(f"{path}: CV features differ from the final features")
        selection = report["datasets"][group]
        train_hash = selection["training_sha256"]
        best_mean = selection["best_mean_f1"]
        fold_strategy = selection["fold_strategy"]
        cv_training = train_path
        cv_features = features
    if selection["selection_split"] != f"{group}:train" or sha256(cv_training.read_bytes()).hexdigest() != train_hash:
        raise ValueError(f"{path}: CV training data differ from the recorded selection inputs")
    return {
        "report": relative(path), "split": selection["selection_split"], "fold_strategy": fold_strategy,
        "parameters": selection["best_parameters"], "mean_f1": best_mean,
        "selection_training_file": relative(cv_training), "selection_features": cv_features,
    }


def read_m1_selection(dataset, train_path):
    train_path = Path(train_path).resolve()
    report = json.loads(RESULTS.read_text(encoding="utf-8"))
    if report["status"] != "complete":
        raise ValueError(f"{RESULTS}: the official baseline run must be complete")
    source = report["groups"][dataset]
    config = source["m1"]
    if (
        source["training_file"] != relative(train_path)
        or config.get("selection", {}).get("training_sha256") != sha256(train_path.read_bytes()).hexdigest()
        or config.get("tol") != m1_logistic.TOL
        or config["max_iterations"] != m1_logistic.MAX_ITER
        or config["solver"] != m1_logistic.SOLVER
    ):
        raise ValueError(f"{RESULTS}: M1 selection uses different data or solver settings")
    return config


def score_target(dataset, rows, paths, subset_path, subset_ids, full):
    gold = {sample_id: row["tags"] for sample_id, row in rows.items()}
    predictions = {model: read_tags_by_id(path) for model, path in paths.items()}
    label_space = "target native labels"
    if dataset == "btc":
        maps = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["transfer_three_types"]["maps"]
        gold = project_tags(gold, maps[dataset])
        predictions = {model: project_tags(answer, maps["conll2003"]) for model, answer in predictions.items()}
        label_space = "shared PER, LOC, ORG; CoNLL-2003 MISC is omitted"
    types = sorted({tag[2:] for tags in gold.values() for tag in tags if tag != "O"}
                   | {tag[2:] for answer in predictions.values() for tags in answer.values() for tag in tags if tag != "O"})

    scores = {"label_space": label_space, "types": types, "subset_ids": relative(subset_path)}
    if full:
        scores["full"] = score_baselines(gold, predictions, token_accuracy=True, entity_types=types)
    subset_gold = {sample_id: gold[sample_id] for sample_id in subset_ids}
    subset_predictions = {
        model: {sample_id: answer[sample_id] for sample_id in subset_ids}
        for model, answer in predictions.items()
    }
    scores["test_300"] = score_baselines(subset_gold, subset_predictions, entity_types=types)
    return scores


def write_answers(stage, targets, dictionary, vectorizer, logistic, crf):
    files = {}
    repairs = {}
    for dataset, rows in targets.items():
        paths = {model: stage / f"{dataset}_{model}.jsonl" for model in MODELS}
        m0_dictionary.write_predictions(paths["m0"], rows, dictionary)
        m1_bio = m1_logistic.write_predictions(paths["m1"], rows, vectorizer, logistic)
        m2_bio = m2_crf.write_predictions(paths["m2"], rows, crf)
        files[dataset] = paths
        repairs[dataset] = {
            model: {"tokens": tokens, "sentences": sentences}
            for model, (tokens, sentences) in zip(MODELS, ((0, 0), m1_bio, m2_bio), strict=True)
        }
    return files, repairs


def run_group(group):
    source, shared_pos, targets = GROUPS[group]
    target_groups = {group: targets}
    if group == "conll2003":
        target_groups["transfer"] = GROUPS["transfer"][2]
    train_path, train_rows = load_rows(source, "train", shared_pos)
    features = {
        "pos_model": POS_MODEL if shared_pos else "original CoNLL-2003 POS",
        "use_chunk_tags": not shared_pos,
        "context_window": 1,
    }
    selection = m2_selection(group, train_path, features)
    print(f"[{group}] training on {len(train_rows):,} {source} sentences", flush=True)
    training = list(train_rows.values())
    dictionary, ambiguous = m0_dictionary.build_dictionary(training)
    validation_path, validation_rows = load_rows(source, "validation", shared_pos)
    vectorizer, logistic, candidates = m1_logistic.select_model(training, list(validation_rows.values()))
    target_rows = {}
    subset_info = {}
    for dataset in (dataset for members in target_groups.values() for dataset in members):
        subset_path = SUBSETS / dataset / "test_300.txt"
        sample_ids = read_subset_ids(subset_path)
        _, rows = load_rows(dataset, "test", shared_pos)
        if len(sample_ids) != 300 or set(sample_ids) - rows.keys():
            raise ValueError(f"{subset_path}: expected 300 unique IDs from the test split")
        target_rows[dataset] = rows if dataset == source else {sample_id: rows[sample_id] for sample_id in sample_ids}
        subset_info[dataset] = (subset_path, sample_ids)

    metadata = {
        "source_dataset": source, "training_file": relative(train_path), "training_records": len(training),
        "m0": {"dictionary_entries": len(dictionary), "ambiguous_names_excluded": ambiguous},
        "m1_m2_features": features,
        "m1": {
            "c": logistic.C, "solver": logistic.solver, "tol": logistic.tol,
            "iterations": int(logistic.n_iter_[0]), "max_iterations": logistic.max_iter,
            "selection": {
                "split": f"{source}:validation", "validation_file": relative(validation_path),
                "training_sha256": sha256(train_path.read_bytes()).hexdigest(),
                "validation_sha256": sha256(validation_path.read_bytes()).hexdigest(),
                "metric": "Exact entity-level micro F1 after BIO repair; ties favor the smaller C.",
                "candidates": candidates,
            },
        },
        "m2": selection,
    }
    results = {name: {**metadata, "targets": {}} for name in target_groups}
    with TemporaryDirectory(dir=RESULTS.parent) as directory:
        stage = Path(directory)
        crf = m2_crf.train_model(training, stage / "model.crfsuite", **selection["parameters"])
        try:
            files, repairs = write_answers(
                stage, {dataset: list(rows.values()) for dataset, rows in target_rows.items()},
                dictionary, vectorizer, logistic, crf,
            )
        finally:
            crf.tagger_.close()

        for dataset, rows in target_rows.items():
            full = dataset == source
            scored = score_target(dataset, rows, files[dataset], *subset_info[dataset], full)
            scored["predicted_records"] = len(rows)
            scored["bio_repairs"] = repairs[dataset]
            scored["bio_repair_scope"] = "full" if full else "test_300"
            scored["gold_file"] = relative(DATA / dataset / "test.jsonl")
            scored["predictions"] = {
                model: relative(PROJECT_DIR / "preds" / dataset / f"{model}_baseline" / "0.jsonl")
                for model in MODELS
            }
            results[group if full else "transfer"]["targets"][dataset] = scored

        for name, result in results.items():
            for dataset, scored in result["targets"].items():
                for model, filename in scored["predictions"].items():
                    path = PROJECT_DIR / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    files[dataset][model].replace(path)
                scores = scored[scored["bio_repair_scope"]]["models"]
                summary = ", ".join(f"{model.upper()}={scores[model]['f1']:.2%}" for model in MODELS)
                print(f"[{name}] {dataset} {scored['bio_repair_scope']}: {summary}", flush=True)
    return results


def save_report(report):
    with TemporaryDirectory(dir=RESULTS.parent) as directory:
        staged = Path(directory) / RESULTS.name
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(RESULTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, len(SOURCES) + 1), default=1)
    args = parser.parse_args()
    report = {"metric": "exact entity-level micro F1", "status": "in_progress", "groups": {}}
    save_report(report)
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        jobs = [executor.submit(run_group, group) for group in SOURCES]
        for job in as_completed(jobs):
            report["groups"].update(job.result())
            save_report(report)
    report["status"] = "complete"
    save_report(report)
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
