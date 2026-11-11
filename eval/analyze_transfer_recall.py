"""Break down M2's existing test recall by entity familiarity and letter case."""

import json
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import bio_to_spans, read_tags_by_id
from eval.score_transfer import MAPPING_PATH, project_tags


PROJECT = Path(__file__).resolve().parents[1]
TRANSFER_REPORT = PROJECT / "results" / "m2_pos_transfer.json"
RESULTS = PROJECT / "results" / "m2_transfer_recall_groups.json"
BTC_RAW = PROJECT / "data" / "raw" / "btc" / "f.json"
DATASETS = ("conll2003", "btc", "wnut17")
CASES = ("all_lowercase", "has_uppercase", "no_cased_letters")
ACCOUNT_NAME = re.compile(r"[A-Za-z0-9_]+")
EMAIL = re.compile(r"[\w.+-]+(?P<at>@)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            yield json.loads(line)


def name_key(tokens):
    return tuple(token.casefold() for token in tokens)


def casing(tokens):
    name = " ".join(tokens)
    if name.islower():
        return "all_lowercase"
    if any(letter.isupper() or letter.istitle() for letter in name):
        return "has_uppercase"
    return "no_cased_letters"


def btc_rows_with_offsets(rows):
    for row, raw in zip(rows, read_jsonl(BTC_RAW), strict=True):
        # Match BTC's preprocessing rule for blank tokens before using raw offsets.
        kept = [
            (word, tag, offset)
            for word, tag, offset in zip(raw["tokens"], raw["entities"], raw["annotation_offsets"], strict=True)
            if word.strip() or tag != "O"
        ]
        if [(word, tag) for word, tag, _ in kept] != list(zip(row["tokens"], row["tags"], strict=True)):
            raise ValueError(f"{row['sample_id']}: BTC tokens or tags differ from the raw release")
        email_markers = {match.start("at") for match in EMAIL.finditer(raw["text"])}
        yield row, [offset for _, _, offset in kept], email_markers


def btc_kind(tokens, offsets, email_markers, start, end):
    first = tokens[start]
    if first == "@" and end == start + 1:
        return "standalone_at"
    if first.startswith("@") and ACCOUNT_NAME.fullmatch(first[1:]) and offsets[start][0] not in email_markers:
        return "account"

    # A separated @ belongs to a handle only when the raw offsets touch.
    username = start + 1 if first == "@" else start
    marker = username - 1
    if (
        marker >= 0
        and tokens[marker] == "@"
        and offsets[marker][0] not in email_markers
        and offsets[marker][1] == offsets[username][0]
        and ACCOUNT_NAME.fullmatch(tokens[username])
    ):
        return "account"
    return "ordinary"


def stats(total, correct):
    return {"gold_entities": total, "correct": correct, "recall": correct / total if total else None}


def summarize_dataset(dataset, target, maps, source, seen_names):
    gold_path = PROJECT / target["gold"]
    gold = project_tags(read_tags_by_id(gold_path), maps[dataset])
    predictions = project_tags(read_tags_by_id(PROJECT / target["predictions"]), maps[source])
    if gold.keys() != predictions.keys():
        raise ValueError(f"{dataset}: test IDs and prediction IDs differ")

    rows = read_jsonl(gold_path)
    paired = btc_rows_with_offsets(rows) if dataset == "btc" else ((row, None, ()) for row in rows)
    totals = Counter()
    matches = Counter()
    for row, offsets, email_markers in paired:
        tokens, sample_id = row["tokens"], row["sample_id"]
        if len(tokens) != len(gold[sample_id]) or len(tokens) != len(predictions[sample_id]):
            raise ValueError(f"{sample_id}: tokens and tags differ in length")
        predicted_spans = bio_to_spans(predictions[sample_id])
        for span in bio_to_spans(gold[sample_id]):
            start, end, _ = span
            name = tokens[start:end]
            kind = btc_kind(tokens, offsets, email_markers, start, end) if offsets is not None else "ordinary"
            group = (kind, name_key(name) in seen_names, casing(name))
            totals[group] += 1
            matches[group] += span in predicted_spans

    total, correct = sum(totals.values()), sum(matches.values())
    official = target["scores"]
    if (total, correct) != (official["tp"] + official["fn"], official["tp"]):
        raise ValueError(f"{dataset}: group counts differ from the official test scores")

    groups = {}
    kinds = ("ordinary", "account", "standalone_at") if dataset == "btc" else ("ordinary",)
    for kind in kinds:
        keys = [(kind, seen, case) for seen in (True, False) for case in CASES]
        breakdown = []
        for key in keys:
            _, seen, case = key
            if totals[key] or (kind != "standalone_at" and case != "no_cased_letters"):
                breakdown.append({"seen_in_training": seen, "casing": case, **stats(totals[key], matches[key])})
        groups[kind] = {
            "total": stats(sum(totals[key] for key in keys), sum(matches[key] for key in keys)),
            "by_familiarity_and_case": breakdown,
        }

    return {
        "gold": target["gold"],
        "predictions": target["predictions"],
        "total": stats(total, correct),
        "groups": groups,
    }


def main():
    transfer = json.loads(TRANSFER_REPORT.read_text(encoding="utf-8"))
    source = transfer["source_dataset"]
    train_path = PROJECT / transfer["training_input"]
    seen_names = {
        name_key(row["tokens"][start:end])
        for row in read_jsonl(train_path)
        for start, end, _ in bio_to_spans(row["tags"])
    }
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["transfer_three_types"]
    datasets = {}
    for dataset in DATASETS:
        result = summarize_dataset(dataset, transfer["targets"][dataset], mapping["maps"], source, seen_names)
        datasets[dataset] = result
        for kind, group in result["groups"].items():
            total = group["total"]
            recall = "n/a" if total["recall"] is None else f"{total['recall']:.2%}"
            print(f"{dataset} / {kind}: {total['correct']:,}/{total['gold_entities']:,} = {recall}")

    report = {
        "training_reference": transfer["training_input"],
        "prediction_report": TRANSFER_REPORT.relative_to(PROJECT).as_posix(),
        "entity_types": mapping["types"],
        "seen_definition": "Same complete case-folded token sequence as any gold training entity, regardless of type.",
        "btc_partition": (
            "Standalone @ spans; spans starting with a handle or its name, checked against raw offsets "
            "and excluding email addresses from the handle group; all other spans."
        ),
        "recall_definition": "Exact boundary and mapped type matches divided by gold entities. Null means no gold entities.",
        "datasets": datasets,
    }
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        staged = Path(staging) / RESULTS.name
        staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        staged.replace(RESULTS)
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
