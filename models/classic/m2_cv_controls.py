"""Compare M2 context windows and inspect document overlap within training CV."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from statistics import mean, pstdev
from tempfile import TemporaryDirectory

from sklearn.model_selection import KFold
from sklearn_crfsuite import CRF

from eval.score_ner import read_tags_by_id, score_ner
from models.classic.features import sentence_features
from models.classic.m2_cross_validate import (
    N_FOLDS, PROJECT_DIR, TRAIN_PATH, VALIDATION_PATH, make_folds, read_sentences,
)
from models.classic.m2_feature_ablation import (
    CRF_OPTIONS, PARAMETERS, predict_feature_rows, validate_grid_reference,
)
from models.classic.m2_grid_search import GRID_PREDICTIONS, GRID_REPORT
from models.classic.m2_native_cv import DATASETS as NATIVE_DATASETS
from models.classic.m2_native_cv import RESULTS as NATIVE_CV
from models.classic.m2_native_cv import folds_for
from models.classic.m2_pos_validation import POS_MODEL, POS_ROOT, load_pos_rows


RESULTS = PROJECT_DIR / "results" / "m2_cv_controls.json"
RANDOM_SEED = 42


def wider_features(row):
    features = sentence_features(row)
    words = [token.lower() for token in row["tokens"]]
    for i, current in enumerate(features):
        if i >= 2:
            current["G5:prev2_lower"] = words[i - 2]
        if i + 2 < len(words):
            current["G5:next2_lower"] = words[i + 2]
    return features


def fit_and_score(training, heldout, model_path, extract, parameters):
    classifier = CRF(**parameters, **CRF_OPTIONS, model_filename=str(model_path))
    classifier.fit((extract(row) for row in training), [row["tags"] for row in training])
    try:
        predictions, bio = predict_feature_rows(heldout, classifier, extract)
    finally:
        classifier.tagger_.close()
    gold = {row["sample_id"]: row["tags"] for row in heldout}
    return {**score_ner(gold, predictions), **bio}


def evaluate_fold(dataset, window, fold, fit_indices, heldout_indices, model_path, parameters):
    rows = (
        list(read_sentences(TRAIN_PATH, "train")) if dataset == "conll2003"
        else load_pos_rows(POS_ROOT / dataset / "train.jsonl", "train", dataset=dataset)
    )
    training = [rows[i] for i in fit_indices]
    heldout = [rows[i] for i in heldout_indices]
    extract = wider_features if window == 2 else sentence_features
    scores = fit_and_score(training, heldout, model_path, extract, parameters)
    print(f"{model_path.stem} fold {fold}: {scores['f1']:.4%}", flush=True)
    return {"fold": fold, **scores}


def fold_membership(rows, folds):
    membership = []
    for fold, (fit, heldout) in enumerate(folds, 1):
        fit_docs = {rows[i]["doc_id"] for i in fit}
        heldout_docs = {rows[i]["doc_id"] for i in heldout}
        seen_sentences = sum(rows[i]["doc_id"] in fit_docs for i in heldout)
        membership.append({
            "fold": fold,
            "train_sentences": len(fit),
            "heldout_sentences": len(heldout),
            "train_documents": len(fit_docs),
            "heldout_documents": len(heldout_docs),
            "overlapping_documents": len(fit_docs & heldout_docs),
            "heldout_sentences_from_overlapping_documents": seen_sentences,
            "heldout_fraction_from_overlapping_documents": seen_sentences / len(heldout),
            "heldout_sample_ids_sha256": sha256(
                "\n".join(rows[i]["sample_id"] for i in heldout).encode("utf-8")
            ).hexdigest(),
        })
    return membership


def summary(scores):
    ordered = sorted(scores, key=lambda score: score["fold"])
    values = [row["f1"] for row in ordered]
    return {"mean_f1": mean(values), "std_f1": pstdev(values), "fold_scores": ordered}


def compare_windows(narrow, wide):
    paired_deltas = [
        {"fold": one["fold"], "f1_delta_pp": 100 * (two["f1"] - one["f1"])}
        for one, two in zip(narrow["fold_scores"], wide["fold_scores"], strict=True)
    ]
    delta = 100 * (wide["mean_f1"] - narrow["mean_f1"])
    return {
        "window_2_minus_window_1_pp": delta,
        "paired_window_deltas_pp": paired_deltas,
        "higher_mean_window": 2 if delta > 0 else 1,
    }


def conll_controls(work_dir, jobs, reuse):
    rows = list(read_sentences(TRAIN_PATH, "train"))
    grouped, fold_sizes = make_folds(rows)
    fingerprint, reference = validate_grid_reference(rows, grouped, fold_sizes)
    if sorted(reference) != list(range(1, N_FOLDS + 1)):
        raise ValueError("the saved grouped reference must contain one score per fold")
    for fold, (_, heldout) in enumerate(grouped, 1):
        score = reference[fold]
        entities = sum(tag.startswith("B-") for i in heldout for tag in rows[i]["tags"])
        if score["scored_records"] != len(heldout) or score["tp"] + score["fn"] != entities:
            raise ValueError(f"grouped reference counts differ in fold {fold}")

    if reuse:
        report = json.loads(RESULTS.read_text(encoding="utf-8"))
        if report["fingerprint"] != fingerprint or report["random_seed"] != RANDOM_SEED:
            raise ValueError("saved CoNLL-2003 controls use different data or parameters")
        return report

    random = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(rows))
    scores = {"grouped_window_1": list(reference.values()), "grouped_window_2": [], "random_window_1": []}
    print("Reusing the verified grouped +/-1 scores and fitting ten training-only models", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                evaluate_fold, "conll2003", window, fold, fit, heldout,
                work_dir / f"{condition}_{fold}.crfsuite", PARAMETERS,
            ): condition
            for condition, window, folds in (("grouped_window_2", 2, grouped), ("random_window_1", 1, random))
            for fold, (fit, heldout) in enumerate(folds, 1)
        }
        for future in as_completed(futures):
            scores[futures[future]].append(future.result())

    results = {condition: summary(values) for condition, values in scores.items()}
    narrow = results["grouped_window_1"]
    wide = results["grouped_window_2"]
    shuffled = results["random_window_1"]
    comparison = compare_windows(narrow, wide)
    return {
        "training_input": TRAIN_PATH.relative_to(PROJECT_DIR).as_posix(),
        "training_records": len(rows),
        "evaluation": "Five held-out folds within CoNLL-2003 training data, scored using native entity labels.",
        "parameters": PARAMETERS,
        "crf_options": CRF_OPTIONS,
        "features": {
            "pos_model": "original CoNLL-2003 POS",
            "use_chunk_tags": True,
            "window_2_change": "Add the lowercase token two positions left or right when it exists.",
        },
        "grouped_reference": GRID_REPORT.relative_to(PROJECT_DIR).as_posix(),
        "fingerprint": fingerprint,
        "random_seed": RANDOM_SEED,
        "fold_membership": {"grouped": fold_membership(rows, grouped), "random": fold_membership(rows, random)},
        "results": results,
        "grouped_window_2_minus_window_1_pp": comparison.pop("window_2_minus_window_1_pp"),
        **comparison,
        "random_minus_grouped_window_1_pp": 100 * (shuffled["mean_f1"] - narrow["mean_f1"]),
        "baseline_configuration": "CoNLL-2003 window and split controls here use original POS and chunks. The shared-POS baseline is a separate setup.",
    }


def conll_dev_check(model_path):
    training = list(read_sentences(TRAIN_PATH, "train"))
    dev = list(read_sentences(VALIDATION_PATH, "validation"))
    gold = {row["sample_id"]: row["tags"] for row in dev}
    if GRID_PREDICTIONS.is_file():
        narrow = score_ner(gold, read_tags_by_id(GRID_PREDICTIONS))
        reference = GRID_PREDICTIONS.relative_to(PROJECT_DIR).as_posix()
    else:
        narrow = fit_and_score(training, dev, model_path.with_name("conll2003_dev_window_1.crfsuite"),
                               sentence_features, PARAMETERS)
        reference = "Refit the existing +/-1 configuration on the full training data."
    wide = fit_and_score(training, dev, model_path, wider_features, PARAMETERS)
    return {
        "selection_rule": "Refit the higher-mean window from grouped training folds and compare on development only.",
        "input": VALIDATION_PATH.relative_to(PROJECT_DIR).as_posix(),
        "window_1_reference_predictions": reference,
        "window_1": narrow,
        "window_2": wide,
        "window_2_minus_window_1_pp": 100 * (wide["f1"] - narrow["f1"]),
    }


def extend_with_native(report, work_dir, jobs):
    saved = json.loads(NATIVE_CV.read_text(encoding="utf-8"))
    features = {"pos_model": POS_MODEL, "use_chunk_tags": False, "context_window": 1}
    if saved["features"] != features:
        raise ValueError("saved native CV used different POS or context features")
    native = {}
    wide_futures = {}
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        dev_future = None
        if report["higher_mean_window"] == 2 and "conll2003_dev_check" not in report:
            dev_future = executor.submit(conll_dev_check, work_dir / "conll2003_dev.crfsuite")
        for dataset in NATIVE_DATASETS:
            folds, metadata = folds_for(dataset)
            reference = saved["datasets"][dataset]
            if any(reference[key] != value for key, value in metadata.items()):
                raise ValueError(f"{dataset}: saved native CV data or folds differ")
            parameters = reference["best_parameters"]
            candidate, = (row for row in reference["candidates"] if all(row[key] == val for key, val in parameters.items()))
            narrow = summary(candidate["fold_scores"])
            if narrow["mean_f1"] != reference["best_mean_f1"] or [
                (row["fold"], row["scored_records"]) for row in narrow["fold_scores"]
            ] != [(row["fold"], row["heldout_sentences"]) for row in metadata["fold_sizes"]]:
                raise ValueError(f"{dataset}: saved native CV scores differ from the reference")
            native[dataset] = {**metadata, "parameters": parameters, "window_1": narrow}
            wide_futures[dataset] = [
                executor.submit(
                    evaluate_fold, dataset, 2, fold, fit, heldout,
                    work_dir / f"{dataset}_window_2_{fold}.crfsuite", parameters,
                ) for fold, (fit, heldout) in enumerate(folds, 1)
            ]

        for dataset, futures in wide_futures.items():
            wide = summary([future.result() for future in futures])
            row = native[dataset]
            row.update({"window_2": wide, **compare_windows(row["window_1"], wide)})
            print(f"{dataset}: window 1 {row['window_1']['mean_f1']:.4%}, window 2 {wide['mean_f1']:.4%}", flush=True)
        if dev_future is not None:
            report["conll2003_dev_check"] = dev_future.result()
    report["native_windows"] = {
        "reference": NATIVE_CV.relative_to(PROJECT_DIR).as_posix(),
        "features": {"pos_model": POS_MODEL, "use_chunk_tags": False},
        "datasets": native,
    }


def add_contiguous_control(report, work_dir, jobs):
    rows = list(read_sentences(TRAIN_PATH, "train"))
    folds = list(KFold(n_splits=N_FOLDS, shuffle=False).split(rows))
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(
                evaluate_fold, "conll2003", 1, fold, fit, heldout,
                work_dir / f"contiguous_window_1_{fold}.crfsuite", PARAMETERS,
            ) for fold, (fit, heldout) in enumerate(folds, 1)
        ]
        scores = summary([future.result() for future in futures])
    report["results"]["contiguous_window_1"] = scores
    report["fold_membership"]["contiguous"] = fold_membership(rows, folds)
    random_mean = report["results"]["random_window_1"]["mean_f1"]
    grouped_mean = report["results"]["grouped_window_1"]["mean_f1"]
    report["random_minus_contiguous_window_1_pp"] = 100 * (random_mean - scores["mean_f1"])
    report["contiguous_minus_grouped_window_1_pp"] = 100 * (scores["mean_f1"] - grouped_mean)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, N_FOLDS + 1), default=3)
    parser.add_argument("--reuse-conll", action="store_true", help="Use saved comparisons and calculate missing controls")
    args = parser.parse_args()
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        work_dir = Path(staging)
        report = conll_controls(work_dir, args.jobs, args.reuse_conll)
        if "native_windows" not in report:
            extend_with_native(report, work_dir, args.jobs)
        if "contiguous_window_1" not in report["results"]:
            add_contiguous_control(report, work_dir, args.jobs)
        staged = work_dir / RESULTS.name
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(RESULTS)

    for condition, result in report["results"].items():
        print(f"{condition}: {result['mean_f1']:.4%} (SD {result['std_f1']:.4%})")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
