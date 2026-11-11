"""Select M2 regularization on document-grouped folds, then score the refit on validation."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import mean, pstdev
from tempfile import TemporaryDirectory

from sklearn.model_selection import GroupKFold

from eval.score_ner import read_tags_by_id, score_ner
from models.classic.m0_dictionary import DATA_DIR, PROJECT_DIR, read_sentences
from models.classic.m2_crf import DEFAULT_OUTPUT as BASELINE_PREDICTIONS
from models.classic.m2_crf import train_model, write_predictions


TRAIN_PATH = DATA_DIR / "train.jsonl"
VALIDATION_PATH = DATA_DIR / "validation.jsonl"
RESULTS_PATH = PROJECT_DIR / "results" / "m2_cv.json"
TUNED_PREDICTIONS = PROJECT_DIR / "preds" / "conll2003" / "m2_cv" / "0.jsonl"
PARAMETER_GRID = ((0.1, 0.1), (0.01, 0.1), (0.1, 0.5), (0.01, 0.5))
EXTRA_RESULTS_PATH = PROJECT_DIR / "results" / "m2_cv_extra.json"
EXTRA_PREDICTIONS = PROJECT_DIR / "preds" / "conll2003" / "m2_cv_extra" / "0.jsonl"
EXTRA_GRID = ((0.001, 0.1), (0.01, 0.01))
N_FOLDS = 5


def make_folds(training):
    doc_ids = [row["doc_id"] for row in training]
    sample_ids = [row["sample_id"] for row in training]
    if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in doc_ids):
        raise ValueError("every training sentence needs a doc_id")
    if len(set(sample_ids)) != len(training):
        raise ValueError("training sample IDs must be unique")

    folds = list(GroupKFold(n_splits=N_FOLDS).split(training, groups=doc_ids))
    fold_sizes = []
    heldout_indices = set()
    for fold, (fit_indices, check_indices) in enumerate(folds, start=1):
        fit_docs = {doc_ids[i] for i in fit_indices}
        check_docs = {doc_ids[i] for i in check_indices}
        if fit_docs & check_docs or heldout_indices.intersection(check_indices):
            raise ValueError(f"fold {fold} shares documents or repeats held-out sentences")
        heldout_indices.update(check_indices)
        fold_sizes.append({
            "fold": fold,
            "train_sentences": len(fit_indices),
            "heldout_sentences": len(check_indices),
            "train_documents": len(fit_docs),
            "heldout_documents": len(check_docs),
        })
    if len(heldout_indices) != len(training):
        raise ValueError("the five held-out folds must cover the training split")
    return folds, fold_sizes


def fit_and_predict(training, heldout, directory, c1, c2):
    classifier = train_model(training, directory / "model.crfsuite", c1=c1, c2=c2)
    predictions = directory / "predictions.jsonl"
    try:
        corrected_tokens, corrected_sentences = write_predictions(predictions, heldout, classifier)
    finally:
        classifier.tagger_.close()
    return predictions, {
        "raw_bio_error_tokens": corrected_tokens,
        "raw_bio_error_sentences": corrected_sentences,
    }


def evaluate_fold(fold, fit_indices, check_indices, work_dir, grid=PARAMETER_GRID):
    # Workers load only train.jsonl and receive indices rather than the entire dataset.
    training = list(read_sentences(TRAIN_PATH, "train"))
    fit_rows = [training[i] for i in fit_indices]
    check_rows = [training[i] for i in check_indices]
    gold = {row["sample_id"]: row["tags"] for row in check_rows}
    scores = []
    for c1, c2 in grid:
        with TemporaryDirectory(dir=work_dir) as staging:
            predictions, bio_counts = fit_and_predict(fit_rows, check_rows, Path(staging), c1, c2)
            result = score_ner(gold, read_tags_by_id(predictions))
        print(f"Fold {fold} | c1={c1:g} c2={c2:g} | F1={result['f1']:.4%}", flush=True)
        scores.append({"fold": fold, **result, **bio_counts})
    return scores


def summarize(fold_scores, grid=PARAMETER_GRID):
    candidates = []
    for index, (c1, c2) in enumerate(grid):
        scores = [fold[index] for fold in fold_scores]
        f1_values = [score["f1"] for score in scores]
        candidates.append({
            "c1": c1,
            "c2": c2,
            "mean_f1": mean(f1_values),
            "std_f1": pstdev(f1_values),
            "fold_scores": scores,
        })
    # max keeps the earlier grid entry if both mean and deviation are equal.
    best = max(candidates, key=lambda result: (result["mean_f1"], -result["std_f1"]))
    return candidates, best


def read_incumbent(fold_sizes):
    report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if (
        report["selection_split"] != "conll2003:train"
        or report["group_key"] != "doc_id"
        or report["context_window"] != 1
        or report["fold_sizes"] != fold_sizes
    ):
        raise ValueError(f"{RESULTS_PATH}: previous settings or fold sizes differ")
    best_parameters = report["best_parameters"]
    for candidate in report["candidates"]:
        if (candidate["c1"], candidate["c2"]) == (best_parameters["c1"], best_parameters["c2"]):
            return candidate
    raise ValueError(f"{RESULTS_PATH}: selected parameters are missing from the candidate scores")


def refit_and_score(training, directory, selected, previous_predictions):
    c1, c2 = selected["c1"], selected["c2"]
    print(f"Refitting the selected c1={c1:g}, c2={c2:g} on all training sentences", flush=True)

    # Open validation only after train-only selection calls for a refit.
    validation = list(read_sentences(VALIDATION_PATH, "validation"))
    gold = {row["sample_id"]: row["tags"] for row in validation}
    previous = score_ner(gold, read_tags_by_id(previous_predictions))
    predictions, bio_counts = fit_and_predict(training, validation, directory, c1, c2)
    refit = score_ner(gold, read_tags_by_id(predictions))
    return predictions, {"previous": previous, "refit": {**refit, **bio_counts}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, N_FOLDS + 1), default=3)
    parser.add_argument(
        "--extend", action="store_true",
        help="Try only the two additional settings against the saved best five-fold mean",
    )
    args = parser.parse_args()

    training = list(read_sentences(TRAIN_PATH, "train"))
    folds, fold_sizes = make_folds(training)
    if args.extend:
        try:
            incumbent = read_incumbent(fold_sizes)
        except (OSError, ValueError, KeyError) as error:
            parser.error(str(error))
        grid = EXTRA_GRID
        results_path = EXTRA_RESULTS_PATH
        output = EXTRA_PREDICTIONS
        previous_predictions = TUNED_PREDICTIONS
    else:
        incumbent = None
        grid = PARAMETER_GRID
        results_path = RESULTS_PATH
        output = TUNED_PREDICTIONS
        previous_predictions = BASELINE_PREDICTIONS
    if not previous_predictions.is_file():
        parser.error(f"previous M2 predictions are missing: {previous_predictions}")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Checking {len(grid)} settings on {N_FOLDS} document-grouped folds", flush=True)
    with TemporaryDirectory(dir=results_path.parent) as staging:
        work_dir = Path(staging)
        jobs = [
            (fold, fit_indices, check_indices, work_dir, grid)
            for fold, (fit_indices, check_indices) in enumerate(folds, start=1)
        ]
        if args.jobs == 1:
            fold_scores = [evaluate_fold(*job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = [executor.submit(evaluate_fold, *job) for job in jobs]
                fold_scores = [future.result() for future in futures]

        candidates, best = summarize(fold_scores, grid)
        for candidate in candidates:
            print(
                f"c1={candidate['c1']:g} c2={candidate['c2']:g} | "
                f"mean F1={candidate['mean_f1']:.4%} +/- {candidate['std_f1']:.4%}",
                flush=True,
            )
        if incumbent is not None:
            best = max(candidates, key=lambda item: item["mean_f1"])
            print(f"Saved best mean F1={incumbent['mean_f1']:.4%}", flush=True)
        refit = incumbent is None or best["mean_f1"] > incumbent["mean_f1"]
        selected = best if refit else incumbent
        results = {
            "selection_split": "conll2003:train",
            "group_key": "doc_id",
            "context_window": 1,
            "fold_sizes": fold_sizes,
            "candidates": candidates,
            "best_parameters": {"c1": selected["c1"], "c2": selected["c2"]},
        }
        if incumbent is not None:
            results["incumbent"] = incumbent
            results["refit_performed"] = refit
        if refit:
            predictions, results["validation"] = refit_and_score(
                training, work_dir, selected, previous_predictions
            )
            output.parent.mkdir(parents=True, exist_ok=True)

        report = work_dir / "results.json"
        report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        if refit:
            predictions.replace(output)
        report.replace(results_path)

    if refit:
        validation = results["validation"]
        print(
            f"Validation F1: previous={validation['previous']['f1']:.4%}, "
            f"refit={validation['refit']['f1']:.4%}"
        )
        print(f"Refit predictions: {output}")
    else:
        print("The saved settings have the highest mean; ending after five-fold comparison.")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
