"""Score a complete news test answer by familiarity with CoNLL-2003 training names."""

import argparse
import json
from pathlib import Path

import numpy as np

from data.make_entity_index import DATA_DIR, OUTPUT_ROOT as INDEX_ROOT, name_key, read_entities
from eval.baseline_statistics import bootstrap_counts, metric_values
from eval.score_m2_familiarity import load_spans, project_groups, score_groups
from eval.score_transfer import MAPPING_PATH


PROJECT = DATA_DIR.parent
REFERENCE = DATA_DIR / "processed" / "conll2003" / "train.jsonl"
GROUPS = ("overall", "seen", "unseen")
FAMILIARITY = (
    "Gold uses the v1 entity index. Predictions use their own full case-folded token sequence "
    "matched against training gold names, irrespective of entity type."
)


def news_settings():
    return json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["conll_temporal_four_types"]


def training_names():
    names = {name_key(tokens) for _, _, _, _, tokens in read_entities("conll2003", "train")}
    if not names:
        raise ValueError(f"{REFERENCE}: no training entity names")
    return names


def f1_intervals(gold, predicted, label_map, sentence_count, *, resamples=1000, seed=42):
    """Resample test sentences and return 95% percentile F1 intervals for each familiarity group."""
    gold_groups, pred_groups = project_groups(gold, label_map), project_groups(predicted, label_map)
    by_sentence = {}
    for group, seen in ((1, True), (2, False)):
        expected, answered = gold_groups[seen], pred_groups[seen]
        for metric, spans in enumerate((expected & answered, answered - expected, expected - answered)):
            for sample_id, *_ in spans:
                by_sentence.setdefault(sample_id, np.zeros((3, 3), dtype=np.int64))[group, metric] += 1
    # Sentences without entities stay in the resampling pool as all-zero rows.
    counts = np.zeros((sentence_count, 3, 3), dtype=np.int64)
    for index, sample_id in enumerate(sorted(by_sentence)):
        counts[index] = by_sentence[sample_id]
    counts[:, 0] = counts[:, 1] + counts[:, 2]
    bounds = np.quantile(metric_values(bootstrap_counts(counts, resamples, seed)), (0.025, 0.975), axis=0)
    return {group: bounds[:, index, 2].tolist() for index, group in enumerate(GROUPS)}


def score_full_test(dataset, prediction_path, names, label_map):
    gold_path = DATA_DIR / "processed" / dataset / "test.jsonl"
    gold, predicted, count = load_spans(
        dataset, {"gold": gold_path, "predictions": prediction_path.resolve()}, names,
    )
    scores = score_groups(gold, predicted, label_map, label_map)
    for group, bounds in f1_intervals(gold, predicted, label_map, count).items():
        scores[group]["f1_ci95"] = bounds
    return {
        "test_records": count,
        "gold": gold_path.relative_to(PROJECT).as_posix(),
        "entity_index": (INDEX_ROOT / dataset / "test_v1.jsonl").relative_to(PROJECT).as_posix(),
        "bootstrap": {"resamples": 1000, "seed": 42, "unit": "sentence", "interval": "95% percentile"},
        **scores,
    }


def main():
    settings = news_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=settings["datasets"])
    parser.add_argument("--pred", required=True, type=Path, help="Full test predictions JSONL")
    args = parser.parse_args()
    label_map = {kind: kind for kind in settings["types"]}
    try:
        scores = score_full_test(args.dataset, args.pred, training_names(), label_map)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps({
        "dataset": args.dataset,
        "entity_types": settings["types"],
        "training_name_reference": REFERENCE.relative_to(PROJECT).as_posix(),
        "familiarity": FAMILIARITY,
        "predictions": str(args.pred),
        **scores,
    }, indent=2))


if __name__ == "__main__":
    main()
