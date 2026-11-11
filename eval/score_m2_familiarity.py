"""Score saved M2 test predictions overall and by training-name familiarity."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from data.make_entity_index import DATA_DIR, OUTPUT_ROOT as INDEX_ROOT, name_key, read_entities
from eval.score_ner import bio_to_spans, read_tags_by_id
from eval.score_transfer import MAPPING_PATH


PROJECT = DATA_DIR.parent
TRANSFER_REPORT = PROJECT / "results" / "m2_pos_transfer.json"
RESULTS = PROJECT / "results" / "m2_seen_unseen_f1.json"


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as source:
        yield from map(json.loads, source)


def load_spans(dataset, target, seen_names):
    index_path = INDEX_ROOT / dataset / "test_v1.jsonl"
    indexed = {}
    for entry in read_jsonl(index_path):
        key = (entry["sample_id"], entry["start"], entry["end"], entry["entity_type"])
        if key in indexed or not isinstance(entry["seen_in_training"], bool):
            raise ValueError(f"{index_path}: invalid or repeated entity {key}")
        indexed[key] = (entry["text"], entry["seen_in_training"])

    predictions = read_tags_by_id(PROJECT / target["predictions"])
    sample_ids = set()
    expected = {}
    predicted = {}
    for row in read_jsonl(PROJECT / target["gold"]):
        sample_id = row["sample_id"]
        if sample_id in sample_ids or sample_id not in predictions:
            raise ValueError(f"{dataset}: duplicate gold ID or missing prediction for {sample_id}")
        sample_ids.add(sample_id)
        tokens, tags, pred_tags = row["tokens"], row["tags"], predictions[sample_id]
        if len(tokens) != len(tags) or len(tokens) != len(pred_tags):
            raise ValueError(f"{sample_id}: tokens and tags differ in length")
        for start, end, kind in bio_to_spans(tags):
            entity = tokens[start:end]
            expected[(sample_id, start, end, kind)] = (" ".join(entity), name_key(entity) in seen_names)
        for start, end, kind in bio_to_spans(pred_tags):
            predicted[(sample_id, start, end, kind)] = name_key(tokens[start:end]) in seen_names

    if not sample_ids or sample_ids != predictions.keys():
        raise ValueError(f"{dataset}: predictions must cover exactly the full test split")
    if indexed != expected:
        raise ValueError(f"{index_path}: entity index does not match gold and the training-name reference")
    return {key: seen for key, (_, seen) in indexed.items()}, predicted, len(sample_ids)


def project_groups(spans, label_map):
    groups = {True: set(), False: set()}
    for (sample_id, start, end, kind), seen in spans.items():
        mapped = label_map[kind]
        if mapped is not None:
            groups[seen].add((sample_id, start, end, mapped))
    return groups


def score_sets(gold, predictions):
    tp = len(gold & predictions)
    fp = len(predictions) - tp
    fn = len(gold) - tp
    return {
        "gold_entities": len(gold), "predicted_entities": len(predictions),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / len(predictions) if predictions else 0.0,
        "recall": tp / len(gold) if gold else 0.0,
        "f1": 2 * tp / (len(gold) + len(predictions)) if gold or predictions else 0.0,
    }


def score_groups(gold, predicted, gold_map, prediction_map):
    gold_groups = project_groups(gold, gold_map)
    pred_groups = project_groups(predicted, prediction_map)
    overall = score_sets(gold_groups[True] | gold_groups[False], pred_groups[True] | pred_groups[False])
    seen = score_sets(gold_groups[True], pred_groups[True])
    unseen = score_sets(gold_groups[False], pred_groups[False])
    if any(seen[k] + unseen[k] != overall[k] for k in ("tp", "fp", "fn")):
        raise ValueError("seen and unseen counts do not add up to the overall counts")
    return {"overall": overall, "seen": seen, "unseen": unseen}


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'Dataset':<12} {'Overall F1':>12} {'Seen F1':>10} {'Unseen F1':>11}")
    for dataset, scores in rows.items():
        print(f"{dataset:<12} {scores['overall']['f1']:>12.2%} "
              f"{scores['seen']['f1']:>10.2%} {scores['unseen']['f1']:>11.2%}")


def main():
    transfer = json.loads(TRANSFER_REPORT.read_text(encoding="utf-8"))
    if transfer["source_dataset"] != "conll2003":
        raise ValueError("v1 entity indices use CoNLL-2003 training as the reference")
    seen_names = {name_key(tokens) for _, _, _, _, tokens in read_entities("conll2003", "train")}
    settings = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    temporal = settings["conll_temporal_four_types"]
    shared = settings["transfer_three_types"]
    native_map = {kind: kind for kind in temporal["types"]}
    shared_prediction_map = shared["maps"][transfer["source_dataset"]]

    native_scores, shared_scores = {}, {}
    for dataset, target in transfer["targets"].items():
        gold, predicted, count = load_spans(dataset, target, seen_names)
        input_files = {
            "test_records": count, "gold": target["gold"], "predictions": target["predictions"],
            "entity_index": (INDEX_ROOT / dataset / "test_v1.jsonl").relative_to(PROJECT).as_posix(),
        }
        if dataset in temporal["datasets"]:
            native_scores[dataset] = {**input_files, **score_groups(gold, predicted, native_map, native_map)}
        grouped = score_groups(gold, predicted, shared["maps"][dataset], shared_prediction_map)
        if any(grouped["overall"][key] != target["scores"][key] for key in ("tp", "fp", "fn")):
            raise ValueError(f"{dataset}: overall shared-type counts differ from the transfer report")
        shared_scores[dataset] = {**input_files, **grouped}

    report = {
        "model_report": TRANSFER_REPORT.relative_to(PROJECT).as_posix(),
        "training_name_reference": "data/processed/conll2003/train.jsonl",
        "matching": "Exact sample ID, token boundaries, and evaluated entity type.",
        "familiarity": "Gold uses v1 indices. Predictions use their own complete case-folded token sequence "
                       "matched against any gold entity type in the training reference.",
        "news_four_types": {"entity_types": temporal["types"], "datasets": native_scores},
        "transfer_three_types": {"entity_types": shared["types"], "datasets": shared_scores},
    }
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        staged = Path(staging) / RESULTS.name
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(RESULTS)
    print_table("News (PER, LOC, ORG, MISC)", native_scores)
    print_table("Transfer (PER, LOC, ORG)", shared_scores)
    print(f"\nResults: {RESULTS}")


if __name__ == "__main__":
    main()
