"""Summarize native NER annotations and cross-dataset name overlap."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.score_ner import bio_to_spans


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUTPUT = ROOT / "results" / "data_exploration.json"
DATASETS = {
    "conll2003": ("CoNLL-2003", "Newswire", "In-domain and transfer source"),
    "wnut17": ("WNUT-17", "Social media", "In-domain and transfer source"),
    "mit_restaurant": ("MIT Restaurant", "Restaurant queries", "In-domain and new entity types"),
    "conllpp": ("CoNLL++", "Newswire", "Temporal and transfer test"),
    "wikigold": ("WikiGold", "Wikipedia", "Transfer test"),
    "btc": ("BTC", "Tweets", "Transfer test"),
}
SPLITS = ("train", "validation", "test")
TRAINING_DATASETS = ("conll2003", "wnut17", "mit_restaurant")
EXAMPLES = {
    "Nationality adjective / cuisine": (("conll2003:test:5", 34, "MISC"), ("mit_restaurant:test:30", 3, "Cuisine")),
    "Company name / product mention": (("btc:test:221", 2, "ORG"), ("wnut17:test:783", 4, "product")),
    "Institution acronym": (("btc:test:1513", 8, "ORG"), ("wnut17:test:234", 8, "group")),
}


def scan(path, dataset, split, example_positions):
    kinds, initial, lowercase = Counter(), Counter(), Counter()
    name_types = defaultdict(set)
    name_mentions = Counter()
    examples = {}
    rows = set()
    tokens = entity_tokens = 0
    with path.open(encoding="utf-8-sig") as source:
        for line in source:
            row = json.loads(line)
            sample_id, words, tags = (row[key] for key in ("sample_id", "tokens", "tags"))
            if (row["dataset"], row["split"]) != (dataset, split) or sample_id in rows or len(words) != len(tags):
                raise ValueError(f"{path}: invalid or duplicate sample {sample_id}")
            rows.add(sample_id)
            tokens += len(words)
            for start, end, kind in bio_to_spans(tags):
                span = words[start:end]
                name = tuple(word.casefold() for word in span)
                entity_tokens += end - start
                kinds[kind] += 1
                name_types[name].add(kind)
                name_mentions[name] += 1
                surface = " ".join(span)
                first_cased = next((char for char in surface if char.lower() != char.upper()), "")
                initial[kind] += bool(first_cased.isupper() or first_cased.istitle())
                lowercase[kind] += surface.islower()
                if (sample_id, start) in example_positions:
                    examples[sample_id, start] = {
                        "sample_id": sample_id, "start": start, "end": end, "entity_type": kind, "entity": surface,
                    }
    entities = sum(kinds.values())
    return {
        "sentences": len(rows), "tokens": tokens, "entities": entities,
        "entity_tokens": entity_tokens,
        "mean_sentence_tokens": tokens / len(rows), "mean_entity_tokens": entity_tokens / entities,
        "type_counts": dict(sorted(kinds.items())),
        "unique_names": len(name_types),
        "initial_uppercase_mentions": sum(initial.values()),
        "all_lowercase_mentions": sum(lowercase.values()),
        "initial_uppercase_fraction": sum(initial.values()) / entities,
        "all_lowercase_fraction": sum(lowercase.values()) / entities,
    }, name_types, name_mentions, examples


def shift(train, test, test_mentions):
    unseen = test.keys() - train.keys()
    shared = test.keys() & train.keys()
    changed_type_set = {name for name in shared if train[name] != test[name]}
    disjoint_types = {name for name in shared if train[name].isdisjoint(test[name])}
    unseen_count = sum(test_mentions[name] for name in unseen)
    return {
        "source": "Tjong Kim Sang and De Meulder (2003), section 2, page 143",
        "source_url": "https://aclanthology.org/W03-0419/",
        "train_and_development_period": "End of August 1996",
        "test_period": "December 1996",
        "test_mentions": sum(test_mentions.values()),
        "unseen_test_mentions": unseen_count,
        "unseen_test_mention_fraction": unseen_count / sum(test_mentions.values()),
        "test_unique_names": len(test),
        "unseen_test_unique_names": len(unseen),
        "shared_train_test_names": len(shared),
        "shared_names_with_different_observed_type_sets": len(changed_type_set),
        "shared_names_with_different_observed_type_sets_fraction": len(changed_type_set) / len(shared),
        "shared_names_with_no_common_type": len(disjoint_types),
        "shared_names_with_no_common_type_fraction": len(disjoint_types) / len(shared),
        "train_names_with_multiple_types": sum(len(types) > 1 for types in train.values()),
        "test_names_with_multiple_types": sum(len(types) > 1 for types in test.values()),
    }


def main():
    positions = {(sample_id, start) for pairs in EXAMPLES.values() for sample_id, start, _ in pairs}
    examples = {}
    datasets, test_names, test_mentions = {}, {}, {}
    for dataset, (name, genre, use) in DATASETS.items():
        splits = {}
        for split in (SPLITS if dataset in TRAINING_DATASETS else ("test",)):
            metrics, types, mentions, found = scan(DATA / dataset / f"{split}.jsonl", dataset, split, positions)
            splits[split] = metrics
            examples.update(found)
            if split == "test":
                test_names[dataset], test_mentions[dataset] = types, mentions
            if dataset == "conll2003" and split == "train":
                conll_train = types
        datasets[dataset] = {"name": name, "genre": genre, "use": use, "splits": splits}

    overlap = {}
    for left, names in test_names.items():
        overlap[left] = {}
        for right, other in test_names.items():
            intersection = len(names.keys() & other.keys())
            union = len(names.keys() | other.keys())
            overlap[left][right] = {"intersection": intersection, "union": union, "jaccard": intersection / union}
    comparisons = {}
    for topic, pairs in EXAMPLES.items():
        references = []
        for sample_id, start, expected_type in pairs:
            evidence = examples[sample_id, start]
            if evidence["entity_type"] != expected_type:
                raise ValueError(f"{sample_id}: expected annotation {expected_type}")
            references.append(evidence)
        if len({row["entity"].casefold() for row in references}) != 1:
            raise ValueError(f"{topic}: examples must share the same case-folded surface form")
        comparisons[topic] = references

    report = {
        "definitions": {
            "analysis_scope": "Native gold annotations. Test splits are used for all comparisons between datasets.",
            "case": "Initial uppercase uses the first cased character. All-lowercase mentions require a cased letter. Both use all mentions as denominator.",
            "overlap": "Jaccard similarity of the unique, complete, case-folded entity token sequences. Labels are ignored.",
            "length": "Tokens in the original corpus tokenization.",
            "examples": "Observed labels for matching surface forms in different local contexts. Offsets are zero-based and end exclusive.",
        },
        "datasets": datasets,
        "test_name_overlap": overlap,
        "conll2003_train_test_shift": shift(conll_train, test_names["conll2003"], test_mentions["conll2003"]),
        "annotation_examples": comparisons,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT.parent) as work:
        staged = Path(work) / OUTPUT.name
        staged.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staged.replace(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
