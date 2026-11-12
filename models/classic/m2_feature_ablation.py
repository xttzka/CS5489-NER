"""Measure each classic CRF feature group's effect on the same five training folds."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev
from tempfile import TemporaryDirectory

from sklearn_crfsuite import CRF

from eval.score_ner import score_ner
from eval.score_transfer import project_tags
from models.classic.features import sentence_features
from models.classic.m1_logistic import repair_iob2
from models.classic.m2_cross_validate import PROJECT_DIR, TRAIN_PATH, make_folds, read_sentences
from models.classic.m2_grid_search import GRID_REPORT, grid_code_hashes


RESULTS = PROJECT_DIR / "results" / "m2_feature_ablation.json"
PARAMETERS = {"c1": 0.0001, "c2": 0.1}
CRF_OPTIONS = {"algorithm": "lbfgs", "max_iterations": 100, "all_possible_transitions": True}
GROUPS = {
    "G1": "word identity and lowercase word",
    "G2": "capitalization, word shape, digit and hyphen flags",
    "G3": "lowercase prefixes and suffixes",
    "G4": "original POS and chunk tags",
    "G5": "lowercase neighbors within one token and sentence boundaries",
}
CONDITIONS = ("full", *(f"without_{group}" for group in GROUPS))
TYPES = ("PER", "LOC", "ORG", "MISC")
TYPE_MAPS = {kind: {label: label if label == kind else None for label in TYPES} for kind in TYPES}


def features(row, condition):
    extracted = sentence_features(row)
    if condition == "full":
        return extracted
    prefix = condition.removeprefix("without_") + ":"
    return [{key: value for key, value in token.items() if not key.startswith(prefix)} for token in extracted]


def fit_feature_model(rows, model_path, extract):
    labels = [row["tags"] for row in rows]
    classifier = CRF(**PARAMETERS, **CRF_OPTIONS, model_filename=str(model_path))
    classifier.fit((extract(row) for row in rows), labels)
    return classifier


def predict_feature_rows(rows, classifier, extract):
    predictions = {}
    corrected_tokens = corrected_sentences = 0
    for row in rows:
        answer, corrections = repair_iob2(classifier.predict_single(extract(row)))
        predictions[row["sample_id"]] = answer
        corrected_tokens += corrections
        corrected_sentences += bool(corrections)
    return predictions, {
        "raw_bio_error_tokens": corrected_tokens,
        "raw_bio_error_sentences": corrected_sentences,
    }


def evaluate_fold(condition, fold, fit_indices, check_indices, directory):
    rows = list(read_sentences(TRAIN_PATH, "train"))
    training = [rows[index] for index in fit_indices]
    heldout = [rows[index] for index in check_indices]
    extract = partial(features, condition=condition)
    with TemporaryDirectory(dir=directory) as staging:
        classifier = fit_feature_model(training, Path(staging) / "model.crfsuite", extract)
        try:
            predictions, bio_counts = predict_feature_rows(heldout, classifier, extract)
        finally:
            classifier.tagger_.close()

    gold = {row["sample_id"]: row["tags"] for row in heldout}
    return {
        "fold": fold,
        **score_ner(gold, predictions),
        "per_type": {
            kind: score_ner(project_tags(gold, mapping), project_tags(predictions, mapping))
            for kind, mapping in TYPE_MAPS.items()
        },
        **bio_counts,
    }


def validate_grid_reference(training, folds, fold_sizes):
    report = json.loads(GRID_REPORT.read_text(encoding="utf-8"))
    fingerprint = report["fingerprint"]
    current = {
        "train_sha256": sha256(TRAIN_PATH.read_bytes()).hexdigest(),
        "heldout_sample_ids_sha256": [
            sha256("\n".join(training[i]["sample_id"] for i in heldout).encode("utf-8")).hexdigest()
            for _, heldout in folds
        ],
        "code_sha256": grid_code_hashes(),
        "library_versions": {name: version(name) for name in fingerprint["library_versions"]},
    }
    if (
        any(fingerprint[key] != value for key, value in current.items())
        or report["fold_sizes"] != fold_sizes
        or report["selection_split"] != "conll2003:train"
        or report["group_key"] != "doc_id"
        or report["context_window"] != 1
        or report["status"] != "complete"
    ):
        raise ValueError("saved grid settings, code, training data or folds differ")
    candidate, = (row for row in report["candidates"] if all(row[k] == v for k, v in PARAMETERS.items()))
    return current, {score["fold"]: score for score in candidate["fold_scores"]}


def summarize(scores):
    summary = {}
    for condition, folds in scores.items():
        values = [fold["f1"] for fold in folds]
        summary[condition] = {
            "mean_f1": mean(values),
            "std_f1": pstdev(values),
            "mean_per_type_f1": {kind: mean(fold["per_type"][kind]["f1"] for fold in folds) for kind in TYPES},
        }
    full = summary["full"]
    for row in summary.values():
        row["mean_f1_delta_vs_full_pp"] = 100 * (row["mean_f1"] - full["mean_f1"])
        row["mean_per_type_f1_delta_vs_full_pp"] = {
            kind: 100 * (row["mean_per_type_f1"][kind] - full["mean_per_type_f1"][kind]) for kind in TYPES
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    training = list(read_sentences(TRAIN_PATH, "train"))
    folds, fold_sizes = make_folds(training)
    fingerprint, reference = validate_grid_reference(training, folds, fold_sizes)
    fingerprint["ablation_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    fingerprint["grid_report_sha256"] = sha256(GRID_REPORT.read_bytes()).hexdigest()
    report = {
        "status": "in_progress",
        "training_input": TRAIN_PATH.relative_to(PROJECT_DIR).as_posix(),
        "group_key": "doc_id",
        "context_window": 1,
        "fold_sizes": fold_sizes,
        "entity_types": TYPES,
        "parameters": PARAMETERS,
        "crf_options": CRF_OPTIONS,
        "feature_groups": GROUPS,
        "full_reference": GRID_REPORT.relative_to(PROJECT_DIR).as_posix(),
        "fingerprint": fingerprint,
        "fold_scores": {condition: [] for condition in CONDITIONS},
    }
    if RESULTS.is_file():
        saved = json.loads(RESULTS.read_text(encoding="utf-8"))
        if saved["fingerprint"] != fingerprint:
            raise ValueError("saved ablation uses different inputs or code")
        report = saved

    completed = {
        (condition, score["fold"])
        for condition, scores in report["fold_scores"].items()
        for score in scores
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        directory = Path(staging)

        def save():
            path = directory / "results.json"
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            path.replace(RESULTS)

        tasks = [
            (condition, fold, fit, check, directory)
            for condition in CONDITIONS
            for fold, (fit, check) in enumerate(folds, start=1)
            if (condition, fold) not in completed
        ]
        if tasks:
            print(f"{len(completed)}/30 folds already scored; starting {len(tasks)} fits", flush=True)
            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {executor.submit(evaluate_fold, *task): task[0] for task in tasks}
                for future in as_completed(futures):
                    condition = futures[future]
                    result = future.result()
                    fold = result["fold"]
                    if condition == "full" and any(result[key] != value for key, value in reference[fold].items()):
                        raise ValueError(f"full-feature fold {fold} does not reproduce the saved grid")
                    scores = report["fold_scores"][condition]
                    scores.append(result)
                    scores.sort(key=lambda item: item["fold"])
                    save()
                    print(f"{condition} | fold {fold} | F1={result['f1']:.4%}", flush=True)

        report["summary"] = summarize(report["fold_scores"])
        report["status"] = "complete"
        save()

    for condition, score in report["summary"].items():
        print(f"{condition:>10}: {score['mean_f1']:.4%} ({score['mean_f1_delta_vs_full_pp']:+.4f} pp)")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
