"""Run CoNLL-2003 CRF tuning through its final grid search in one command."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import read_tags_by_id, score_ner
from models.classic.m2_cross_validate import (
    BASELINE_PREDICTIONS, EXTRA_PREDICTIONS, EXTRA_RESULTS_PATH, N_FOLDS, PROJECT_DIR,
    RESULTS_PATH, TRAIN_PATH, TUNED_PREDICTIONS, VALIDATION_PATH, fit_and_predict, read_sentences,
)
from models.classic.m2_grid_search import GRID_PREDICTIONS, GRID_REPORT, run_grid


def run_stage(module, *options):
    subprocess.run(
        [sys.executable, "-B", "-m", f"models.classic.{module}", *options],
        check=True, cwd=PROJECT_DIR,
    )


def restore_validation_answer(answer, report_path):
    """Recreate an omitted answer from its saved tuning result without rerunning CV."""
    if answer.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    training = list(read_sentences(TRAIN_PATH, "train"))
    validation = list(read_sentences(VALIDATION_PATH, "validation"))
    gold = {row["sample_id"]: row["tags"] for row in validation}
    expected = report["validation"]["refit"]
    parameters = report["best_parameters"]
    answer.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=answer.parent) as temporary:
        predictions, _ = fit_and_predict(training, validation, Path(temporary), **parameters)
        scores = score_ner(gold, read_tags_by_id(predictions))
        if any(scores[key] != expected[key] for key in ("tp", "fp", "fn")):
            raise ValueError(f"{report_path}: refit does not reproduce the saved validation counts")
        predictions.replace(answer)
    print(f"Restored validation answer: {answer}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, N_FOLDS + 1), default=N_FOLDS)
    args = parser.parse_args()
    job_options = ("--jobs", str(args.jobs))

    if not RESULTS_PATH.is_file():
        if not BASELINE_PREDICTIONS.is_file():
            run_stage("m2_crf")
        run_stage("m2_cross_validate", *job_options)
    if not EXTRA_RESULTS_PATH.is_file():
        restore_validation_answer(TUNED_PREDICTIONS, RESULTS_PATH)
        run_stage("m2_cross_validate", "--extend", *job_options)

    if not GRID_REPORT.is_file() or not GRID_PREDICTIONS.is_file():
        extra = json.loads(EXTRA_RESULTS_PATH.read_text(encoding="utf-8"))
        if extra["refit_performed"]:
            restore_validation_answer(EXTRA_PREDICTIONS, EXTRA_RESULTS_PATH)
        else:
            restore_validation_answer(TUNED_PREDICTIONS, RESULTS_PATH)
    run_grid(args.jobs)


if __name__ == "__main__":
    main()
