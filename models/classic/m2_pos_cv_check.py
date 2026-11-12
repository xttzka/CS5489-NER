"""Repeat the CoNLL-2003 regularization grid with the shared POS features of the official baseline."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import read_tags_by_id, score_ner
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_cross_validate import N_FOLDS, fit_and_predict, make_folds, summarize
from models.classic.m2_grid_search import C1_VALUES, C2_VALUES
from models.classic.m2_pos_validation import C1, C2, POS_MODEL, POS_ROOT, load_pos_rows, verify_alignment


DATASET = "conll2003"
TRAIN_PATH = POS_ROOT / DATASET / "train.jsonl"
GRID = tuple((c1, c2) for c1 in C1_VALUES for c2 in C2_VALUES)
RESULTS = PROJECT_DIR / "results" / "m2_pos_cv_check.json"


def evaluate_fold(fold, fit_indices, check_indices, directory):
    rows = load_pos_rows(TRAIN_PATH, "train", dataset=DATASET)
    fit = [rows[index] for index in fit_indices]
    heldout = [rows[index] for index in check_indices]
    gold = {row["sample_id"]: row["tags"] for row in heldout}
    scores = []
    for c1, c2 in GRID:
        with TemporaryDirectory(dir=directory) as staging:
            predictions, bio = fit_and_predict(fit, heldout, Path(staging), c1, c2)
            result = score_ner(gold, read_tags_by_id(predictions))
        scores.append({"fold": fold, **result, **bio})
        print(f"fold {fold} | c1={c1:g} c2={c2:g} | F1={result['f1']:.4%}", flush=True)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, N_FOLDS + 1), default=N_FOLDS)
    args = parser.parse_args()

    rows = load_pos_rows(TRAIN_PATH, "train", dataset=DATASET)
    verify_alignment(PROJECT_DIR / "data" / "processed" / DATASET / TRAIN_PATH.name, rows, "train")
    folds, fold_sizes = make_folds(rows)
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        work = Path(staging)
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            jobs = [executor.submit(evaluate_fold, fold, fit, heldout, work)
                    for fold, (fit, heldout) in enumerate(folds, 1)]
            scores = [job.result() for job in jobs]
        candidates, best = summarize(scores, GRID)
        official = next(row for row in candidates if (row["c1"], row["c2"]) == (C1, C2))
        report = {
            "method": "M2 CRF",
            "features": {"pos_model": POS_MODEL, "use_chunk_tags": False, "context_window": 1},
            "training_input": TRAIN_PATH.relative_to(PROJECT_DIR).as_posix(),
            "training_sha256": sha256(TRAIN_PATH.read_bytes()).hexdigest(),
            "selection_split": f"{DATASET}:train",
            "fold_strategy": "Five held-out folds grouped by doc_id.",
            "fold_sizes": fold_sizes,
            "grid": {"c1": list(C1_VALUES), "c2": list(C2_VALUES)},
            "candidates": candidates,
            "best_parameters": {key: best[key] for key in ("c1", "c2")},
            "best_mean_f1": best["mean_f1"],
            "best_std_f1": best["std_f1"],
            "official_parameters": {"c1": C1, "c2": C2},
            "official_mean_f1": official["mean_f1"],
            "official_std_f1": official["std_f1"],
            "best_minus_official_mean_f1_pp": 100 * (best["mean_f1"] - official["mean_f1"]),
        }
        staged = work / RESULTS.name
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(RESULTS)
    print(f"Official c1={C1:g} c2={C2:g}: {official['mean_f1']:.4%} | "
          f"best c1={best['c1']:g} c2={best['c2']:g}: {best['mean_f1']:.4%}")
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
