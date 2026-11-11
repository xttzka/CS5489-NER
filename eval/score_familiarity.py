"""Score a complete news test answer by familiarity with CoNLL-2003 training names."""

import argparse
import json
from pathlib import Path

from data.make_entity_index import DATA_DIR, OUTPUT_ROOT as INDEX_ROOT, name_key, read_entities
from eval.score_m2_familiarity import load_spans, score_groups
from eval.score_transfer import MAPPING_PATH


PROJECT = DATA_DIR.parent
REFERENCE = DATA_DIR / "processed" / "conll2003" / "train.jsonl"
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


def score_full_test(dataset, prediction_path, names, label_map):
    gold_path = DATA_DIR / "processed" / dataset / "test.jsonl"
    gold, predicted, count = load_spans(
        dataset, {"gold": gold_path, "predictions": prediction_path.resolve()}, names,
    )
    return {
        "test_records": count,
        "gold": gold_path.relative_to(PROJECT).as_posix(),
        "entity_index": (INDEX_ROOT / dataset / "test_v1.jsonl").relative_to(PROJECT).as_posix(),
        **score_groups(gold, predicted, label_map, label_map),
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
