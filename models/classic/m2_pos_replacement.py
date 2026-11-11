"""Compare one M2 model on paired original and replaced CoNLL-2003 sentences."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from data.make_entity_replacement import OUTPUT_DIR as PAIR_DIR, VERSION, read_rows
from eval.score_ner import bio_to_spans, read_tags_by_id, score_ner
from models.classic.m0_dictionary import PROJECT_DIR
from models.classic.m2_crf import train_model, write_predictions
from models.classic.m2_pos_validation import C1, C2, POS_DATA, POS_MODEL, PRED_ROOT, load_pos_rows


ENTITY_TYPES = ("PER", "LOC", "ORG", "MISC")


def verify_pairs(original, replaced, changes):
    for before, after, change in zip(original, replaced, changes, strict=True):
        sample_id = before["sample_id"]
        if sample_id != change["sample_id"] or any(before[key] != after[key] for key in before if key != "tokens"):
            raise ValueError(f"{sample_id}: paired metadata or labels differ")
        start, end, kind = change["start"], change["end"], change["entity_type"]
        tokens = before["tokens"]
        if (
            len(tokens) != len(after["tokens"])
            or (start, end, kind) not in bio_to_spans(before["tags"])
            or tokens[start:end] != change["original_tokens"]
            or after["tokens"][start:end] != change["replacement_tokens"]
            or tokens[:start] != after["tokens"][:start]
            or tokens[end:] != after["tokens"][end:]
        ):
            raise ValueError(f"{sample_id}: paired tokens do not match the replacement log")


def summarize_targets(items):
    return {
        "targets": len(items),
        "original_correct": sum(item["original_correct"] for item in items),
        "replaced_correct": sum(item["replaced_correct"] for item in items),
        "both_correct": sum(item["original_correct"] and item["replaced_correct"] for item in items),
        "original_only": sum(item["original_correct"] and not item["replaced_correct"] for item in items),
        "replaced_only": sum(item["replaced_correct"] and not item["original_correct"] for item in items),
        "neither_correct": sum(not item["original_correct"] and not item["replaced_correct"] for item in items),
    }


def score_targets(changes, predictions):
    items = []
    for change in changes:
        sample_id = change["sample_id"]
        span = change["start"], change["end"], change["entity_type"]
        item = {key: change[key] for key in (
            "sample_id", "start", "end", "entity_type", "original_text", "replacement_text",
        )}
        for condition, answer in predictions.items():
            item[f"{condition}_correct"] = span in bio_to_spans(answer[sample_id])
        items.append(item)
    return {
        **summarize_targets(items),
        "per_type": {kind: summarize_targets([item for item in items if item["entity_type"] == kind])
                     for kind in ENTITY_TYPES},
        "items": items,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=(VERSION,), default=VERSION)
    version = parser.parse_args().version
    suffix = f"_{version}"
    original_path = PAIR_DIR / f"original_{version}.jsonl"
    replaced_path = PAIR_DIR / f"replaced_{version}.jsonl"
    changes_path = PAIR_DIR / f"replacements_{version}.jsonl"
    results = PROJECT_DIR / "results" / f"m2_pos_replacement{suffix}.json"

    train_path = POS_DATA / "train.jsonl"
    training = load_pos_rows(train_path, "train")
    original = load_pos_rows(original_path, "test")
    replaced = load_pos_rows(replaced_path, "test")
    changes = list(read_rows(changes_path))
    verify_pairs(original, replaced, changes)
    gold = read_tags_by_id(original_path)

    results.parent.mkdir(parents=True, exist_ok=True)
    conditions, answers, pending = {}, {}, {}
    with TemporaryDirectory(dir=results.parent) as staging:
        work_dir = Path(staging)
        classifier = train_model(training, work_dir / "model.crfsuite", c1=C1, c2=C2)
        try:
            for condition, input_path, rows in (("original", original_path, original), ("replaced", replaced_path, replaced)):
                staged = work_dir / f"{condition}.jsonl"
                repairs, repaired_sentences = write_predictions(staged, rows, classifier)
                answers[condition] = read_tags_by_id(staged)
                scores = score_ner(gold, answers[condition])
                output = PRED_ROOT / f"m2_pos_replacement{suffix}_{condition}" / "0.jsonl"
                pending[staged] = output
                conditions[condition] = {
                    "input": input_path.relative_to(PROJECT_DIR).as_posix(),
                    "predictions": output.relative_to(PROJECT_DIR).as_posix(),
                    "scores": scores,
                    "raw_bio_error_tokens": repairs,
                    "raw_bio_error_sentences": repaired_sentences,
                }
                print(f"{condition}: four-type F1={scores['f1']:.4%}", flush=True)
        finally:
            classifier.tagger_.close()

        targets = score_targets(changes, answers)
        report = {
            "method": "M2 CRF",
            "parameters": {"c1": C1, "c2": C2},
            "features": {"pos_model": POS_MODEL, "use_chunk_tags": False},
            "training_input": train_path.relative_to(PROJECT_DIR).as_posix(),
            "training_records": len(training),
            "paired_test_sentences": len(changes),
            "entity_types": ENTITY_TYPES,
            "replacement_log": changes_path.relative_to(PROJECT_DIR).as_posix(),
            "conditions": conditions,
            "f1_delta_replaced_minus_original_pp": (
                conditions["replaced"]["scores"]["f1"] - conditions["original"]["scores"]["f1"]
            ) * 100,
            "replacement_targets": targets,
        }
        staged_report = work_dir / "results.json"
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for staged, output in pending.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(output)
        staged_report.replace(results)

    print(f"Exact target matches: {targets['original_correct']}/{targets['targets']} before, "
          f"{targets['replaced_correct']}/{targets['targets']} after")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
