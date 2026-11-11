"""Extend M2's train-only grid while reusing completed five-fold results."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from models.classic.m2_cross_validate import (
    EXTRA_PREDICTIONS,
    EXTRA_RESULTS_PATH,
    N_FOLDS,
    PROJECT_DIR,
    RESULTS_PATH,
    TRAIN_PATH,
    TUNED_PREDICTIONS,
    evaluate_fold,
    make_folds,
    read_sentences,
    refit_and_score,
    summarize,
)


C1_VALUES = (0, 0.0001, 0.0005, 0.001, 0.01)
C2_VALUES = (0.01, 0.03, 0.1, 0.5)
GRID = tuple((c1, c2) for c1 in C1_VALUES for c2 in C2_VALUES)
GRID_REPORT = RESULTS_PATH.with_name("m2_cv_grid.json")
GRID_PREDICTIONS = EXTRA_PREDICTIONS.parent.with_name("m2_cv_grid") / "0.jsonl"


def parameter_pair(candidate):
    return candidate["c1"], candidate["c2"]


def check_report(report, path, fold_settings, gold_counts):
    if any(report[key] != value for key, value in fold_settings.items()):
        raise ValueError(f"{path}: stored settings or fold sizes differ")
    for candidate in report["candidates"]:
        scores = candidate["fold_scores"]
        if len(scores) != N_FOLDS or any(
            score["fold"] != size["fold"]
            or score["gold_records"] != size["heldout_sentences"]
            or score["scored_records"] != size["heldout_sentences"]
            or score["tp"] + score["fn"] != gold_count
            for score, size, gold_count in zip(scores, fold_settings["fold_sizes"], gold_counts)
        ):
            raise ValueError(f"{path}: stored fold counts differ for {parameter_pair(candidate)}")
        if candidate["mean_f1"] != mean(score["f1"] for score in scores):
            raise ValueError(f"{path}: inconsistent mean F1 for {parameter_pair(candidate)}")


def run_grid(jobs):
    training = list(read_sentences(TRAIN_PATH, "train"))
    folds, fold_sizes = make_folds(training)
    fold_settings = {
        "selection_split": "conll2003:train",
        "group_key": "doc_id",
        "context_window": 1,
        "fold_sizes": fold_sizes,
    }
    heldout_rows = [[training[i] for i in indices] for _, indices in folds]
    gold_counts = [sum(tag.startswith("B-") for row in rows for tag in row["tags"]) for rows in heldout_rows]

    cached = {}
    history = {}
    for path in (RESULTS_PATH, EXTRA_RESULTS_PATH):
        raw_report = path.read_bytes()
        history[path.name] = sha256(raw_report).hexdigest()
        previous_report = json.loads(raw_report)
        check_report(previous_report, path, fold_settings, gold_counts)
        for candidate in previous_report["candidates"]:
            pair = parameter_pair(candidate)
            if pair in cached:
                raise ValueError(f"duplicate stored scores for {pair}")
            cached[pair] = {**candidate, "source": path.name}
    incumbent = cached[parameter_pair(previous_report["best_parameters"])]
    previous_predictions = EXTRA_PREDICTIONS if previous_report["refit_performed"] else TUNED_PREDICTIONS

    code_paths = (
        "models/classic/features.py",
        "models/classic/m0_dictionary.py",
        "models/classic/m1_logistic.py",
        "models/classic/m2_crf.py",
        "models/classic/m2_cross_validate.py",
        "models/classic/m2_grid_search.py",
        "eval/score_ner.py",
    )
    fingerprint = {
        "train_sha256": sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "heldout_sample_ids_sha256": [
            sha256("\n".join(row["sample_id"] for row in rows).encode("utf-8")).hexdigest()
            for rows in heldout_rows
        ],
        "history_sha256": history,
        "code_sha256": {
            path: sha256((PROJECT_DIR / path).read_bytes()).hexdigest() for path in code_paths
        },
        "library_versions": {
            name: version(name) for name in ("scikit-learn", "sklearn-crfsuite", "python-crfsuite")
        },
    }
    candidates = {pair: cached[pair] for pair in GRID if pair in cached}
    if GRID_REPORT.is_file():
        saved = json.loads(GRID_REPORT.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise ValueError(f"{GRID_REPORT}: checkpoint inputs differ from the current run")
        check_report(saved, GRID_REPORT, fold_settings, gold_counts)
        candidates.update((parameter_pair(candidate), candidate) for candidate in saved["candidates"])
        if saved["status"] == "complete" and len(candidates) == len(GRID) and (
            not saved["refit_performed"] or GRID_PREDICTIONS.is_file()
        ):
            print(f"Grid search already complete: {GRID_REPORT}")
            return

    print(f"{len(GRID)} grid settings | {len(candidates)} reused | {len(GRID) - len(candidates)} to run", flush=True)
    report = {
        **fold_settings,
        "fingerprint": fingerprint,
        "grid": {"c1": C1_VALUES, "c2": C2_VALUES},
        "incumbent": incumbent,
        "historical_outside_grid": [candidate for pair, candidate in cached.items() if pair not in GRID],
        "status": "in_progress",
    }
    with TemporaryDirectory(dir=GRID_REPORT.parent) as staging:
        work_dir = Path(staging)

        def save_report():
            report["candidates"] = [candidates[pair] for pair in GRID if pair in candidates]
            staged = work_dir / "results.json"
            staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            staged.replace(GRID_REPORT)

        pool = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else nullcontext()
        with pool as executor:
            for c1, c2 in GRID:
                pair = (c1, c2)
                if pair in candidates:
                    continue
                fold_jobs = [(i, fit, check, work_dir, (pair,)) for i, (fit, check) in enumerate(folds, 1)]
                if executor is None:
                    scores = [evaluate_fold(*job) for job in fold_jobs]
                else:
                    futures = [executor.submit(evaluate_fold, *job) for job in fold_jobs]
                    scores = [future.result() for future in futures]
                _, candidate = summarize(scores, (pair,))
                candidates[pair] = {**candidate, "source": "new_grid_run"}
                save_report()
                print(
                    f"{len(candidates)}/{len(GRID)} complete | c1={c1:g} c2={c2:g} | "
                    f"mean F1={candidate['mean_f1']:.4%}", flush=True,
                )

        best = max((candidates[pair] for pair in GRID), key=lambda item: item["mean_f1"])
        refit = best["mean_f1"] > incumbent["mean_f1"]
        selected = best if refit else incumbent
        report["best_parameters"] = {key: selected[key] for key in ("c1", "c2")}
        report["refit_performed"] = refit
        if refit:
            predictions, report["validation"] = refit_and_score(training, work_dir, selected, previous_predictions)
            GRID_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
            predictions.replace(GRID_PREDICTIONS)
            report["validation_predictions"] = GRID_PREDICTIONS.relative_to(PROJECT_DIR).as_posix()

        report["status"] = "complete"
        save_report()

    print(f"Selected c1={selected['c1']:g} c2={selected['c2']:g} | mean F1={selected['mean_f1']:.4%}")
    if refit:
        validation = report["validation"]
        print(f"Validation F1: previous={validation['previous']['f1']:.4%}, refit={validation['refit']['f1']:.4%}")
    print(f"Results: {GRID_REPORT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, N_FOLDS + 1), default=5)
    args = parser.parse_args()
    run_grid(args.jobs)


if __name__ == "__main__":
    main()
