"""Tune native-label M2 baselines with five contiguous blocks of training data."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.model_selection import KFold

from eval.score_ner import read_tags_by_id, score_ner
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_cross_validate import N_FOLDS, fit_and_predict, summarize
from models.classic.m2_pos_validation import POS_MODEL, POS_ROOT, load_pos_rows, verify_alignment


DATASETS = ("wnut17", "mit_restaurant")
GRID = ((0.0001, 0.1), (0.001, 0.1), (0.01, 0.1), (0.1, 0.1), (0.01, 0.5))
RESULTS = PROJECT_DIR / "results" / "m2_native_cv.json"


def evaluate_fold(dataset, fold, fit_indices, check_indices, directory):
    train_path = POS_ROOT / dataset / "train.jsonl"
    rows = load_pos_rows(train_path, "train", dataset=dataset)
    fit = [rows[index] for index in fit_indices]
    heldout = [rows[index] for index in check_indices]
    gold = {row["sample_id"]: row["tags"] for row in heldout}
    scores = []
    for c1, c2 in GRID:
        with TemporaryDirectory(dir=directory) as staging:
            predictions, bio = fit_and_predict(fit, heldout, Path(staging), c1, c2)
            result = score_ner(gold, read_tags_by_id(predictions))
        scores.append({"fold": fold, **result, **bio})
        print(f"{dataset} fold {fold} | c1={c1:g} c2={c2:g} | F1={result['f1']:.4%}", flush=True)
    return scores


def folds_for(dataset):
    train_path = POS_ROOT / dataset / "train.jsonl"
    rows = load_pos_rows(train_path, "train", dataset=dataset)
    verify_alignment(PROJECT_DIR / "data" / "processed" / dataset / train_path.name, rows, "train")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"{train_path}: training sample IDs must be unique")
    if any(row.get("doc_id") is not None for row in rows):
        raise ValueError(f"{train_path}: expected no document IDs for contiguous block CV")

    folds = list(KFold(n_splits=N_FOLDS, shuffle=False).split(rows))
    summary = {
        "training_input": train_path.relative_to(PROJECT_DIR).as_posix(),
        "training_sha256": sha256(train_path.read_bytes()).hexdigest(),
        "training_records": len(rows),
        "selection_split": f"{dataset}:train",
        "fold_strategy": "Five contiguous, nearly equal held-out blocks in original training order.",
        "fold_sizes": [
            {
                "fold": fold,
                "train_sentences": len(fit),
                "heldout_sentences": len(heldout),
                "first_heldout_id": rows[heldout[0]]["sample_id"],
                "last_heldout_id": rows[heldout[-1]]["sample_id"],
            }
            for fold, (fit, heldout) in enumerate(folds, 1)
        ],
    }
    return folds, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, N_FOLDS + 1), default=N_FOLDS)
    args = parser.parse_args()

    report = {
        "method": "M2 CRF",
        "features": {"pos_model": POS_MODEL, "use_chunk_tags": False, "context_window": 1},
        "metric": "Mean exact entity-level micro F1 across held-out training folds, native labels.",
        "selection_rule": "Highest unrounded mean F1; exact ties keep the first grid setting.",
        "datasets": {},
    }
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        work_dir = Path(staging)
        jobs = {}
        for dataset in DATASETS:
            folds, summary = folds_for(dataset)
            report["datasets"][dataset] = summary
            jobs[dataset] = [
                (dataset, fold, fit, heldout, work_dir)
                for fold, (fit, heldout) in enumerate(folds, 1)
            ]

        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                dataset: [executor.submit(evaluate_fold, *job) for job in entries]
                for dataset, entries in jobs.items()
            }
            for dataset in DATASETS:
                scores = [future.result() for future in futures[dataset]]
                candidates, _ = summarize(scores, GRID)
                best = max(candidates, key=lambda candidate: candidate["mean_f1"])
                report["datasets"][dataset].update({
                    "candidates": candidates,
                    "best_parameters": {key: best[key] for key in ("c1", "c2")},
                    "best_mean_f1": best["mean_f1"],
                    "best_std_f1": best["std_f1"],
                })
                print(f"{dataset} selected c1={best['c1']:g} c2={best['c2']:g} | mean F1={best['mean_f1']:.4%}", flush=True)

        staged = work_dir / "results.json"
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(RESULTS)

    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
