"""Score exact entity matches in the shared PER, LOC, and ORG label space."""

import argparse
import json
from pathlib import Path

from eval.score_ner import bio_to_spans, read_subset_ids, read_tags_by_id, score_ner


MAPPING_PATH = Path(__file__).resolve().parents[1] / "data" / "ner_label_mapping.json"


def project_tags(records, label_map):
    projected = {}
    for sample_id, tags in records.items():
        try:
            bio_to_spans(tags)
            mapped_tags = []
            for tag in tags:
                if tag == "O":
                    mapped_tags.append(tag)
                    continue
                mapped_type = label_map[tag[2:]]
                mapped_tags.append(f"{tag[:2]}{mapped_type}" if mapped_type else "O")
        except (KeyError, ValueError) as error:
            raise ValueError(f"{sample_id}: cannot map labels ({error})") from error
        projected[sample_id] = mapped_tags
    return projected


def score_transfer(gold, predictions, *, target_dataset, prediction_schema):
    settings = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["transfer_three_types"]
    maps = settings["maps"]
    for dataset in (target_dataset, prediction_schema):
        if maps.get(dataset) is None:
            raise ValueError(f"{dataset!r} has no shared three-type label mapping")

    gold = project_tags(gold, maps[target_dataset])
    predictions = project_tags(predictions, maps[prediction_schema])
    scores = score_ner(gold, predictions)
    scores["per_type"] = {}
    for entity_type in settings["types"]:
        one_type = {label: label if label == entity_type else None for label in settings["types"]}
        scores["per_type"][entity_type] = score_ner(
            project_tags(gold, one_type), project_tags(predictions, one_type)
        )
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--target-dataset", required=True, help="Dataset used for the gold labels")
    parser.add_argument("--prediction-schema", required=True, help="Dataset whose labels the model emits")
    parser.add_argument("--subset-ids", type=Path, help="Require a prediction for exactly these IDs")
    args = parser.parse_args()

    try:
        gold = read_tags_by_id(args.gold)
        if args.subset_ids:
            sample_ids = read_subset_ids(args.subset_ids)
            missing = set(sample_ids) - gold.keys()
            if missing:
                raise ValueError(f"subset ID {min(missing)!r} is absent from {args.gold}")
            gold = {sample_id: gold[sample_id] for sample_id in sample_ids}
        scores = score_transfer(
            gold,
            read_tags_by_id(args.pred),
            target_dataset=args.target_dataset,
            prediction_schema=args.prediction_schema,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
