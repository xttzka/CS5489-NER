"""Shared data access, type descriptions and span alignment for the zero-shot and LLM extractors."""

import json
from pathlib import Path

import random

from eval.score_ner import read_subset_ids, score_ner
from eval.score_transfer import score_transfer


PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data" / "processed"
SUBSETS = PROJECT / "data" / "subsets"
PREDS = PROJECT / "preds"
RAW_OUTPUT = PROJECT / "output" / "llm_raw"
RESULTS = PROJECT / "results" / "c"

NEWS_TYPES = {
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "MISC": "miscellaneous name such as a nationality, event, language or product",
}
TYPE_DESCRIPTIONS = {
    "conll2003": NEWS_TYPES,
    "conllpp": NEWS_TYPES,
    "wikigold": NEWS_TYPES,
    "btc": {key: NEWS_TYPES[key] for key in ("PER", "ORG", "LOC")},
    "wnut17": {
        "person": "person",
        "location": "location",
        "corporation": "company",
        "group": "group such as a band, sports team or organization",
        "product": "product",
        "creative-work": "creative work such as a film, song, book or show",
    },
    "mit_restaurant": {
        "Amenity": "restaurant amenity or feature",
        "Cuisine": "cuisine",
        "Dish": "dish or food item",
        "Hours": "opening hours or time",
        "Location": "location or distance",
        "Price": "price level",
        "Rating": "rating or review quality",
        "Restaurant_Name": "restaurant name",
    },
}
# Few-shot examples and annotation notes come from the corpus that defines the target label scheme.
EXAMPLE_SOURCE = {"conllpp": "conll2003", "wikigold": "conll2003", "btc": "conll2003"}
GUIDELINES = {
    "conll2003": (
        "Nationalities, languages and adjectives made from place names, like German or British, are MISC.",
        "Countries, cities, regions and other geographic or political places are LOC.",
        "Sports teams, companies, political parties, government bodies and news agencies are ORG, "
        "also when the team is named by its city or country.",
        "Events, competitions, wars, product names and titles of works are MISC.",
        "A person entity is the name alone. Titles and roles like President or Mr are left out.",
        "Dates, times, amounts of money, percentages and other numbers are not entities.",
        "An entity is the full name with no article or other surrounding word.",
    ),
    "wnut17": (
        "Tag only names of specific things. Common nouns, pronouns, hashtags used as topics "
        "and user mentions starting with @ are not entities.",
        "corporation is a company name. group covers bands, sports teams and other named groups of people.",
        "product is a named commercial product or device. A company name alone is corporation.",
        "creative-work is the title of a film, show, song, book or game.",
        "An entity is the full name with no article or other surrounding word.",
    ),
}
GUIDELINES["conllpp"] = GUIDELINES["wikigold"] = GUIDELINES["conll2003"]


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_rows(dataset, split, subset=None):
    """Return gold rows of one split, restricted to a fixed ID list when one is named."""
    rows = read_jsonl(DATA / dataset / f"{split}.jsonl")
    if subset is None:
        return rows
    by_id = {row["sample_id"]: row for row in rows}
    return [by_id[sample_id] for sample_id in read_subset_ids(subset_path(dataset, subset))]


def load_input(path):
    """Read a derived test file such as a replacement or perturbation set."""
    return read_jsonl(PROJECT / path)


def sample_train(dataset, size, seed):
    """Draw training sentences: a count when size is an integer string, a share when it has a point."""
    rows = load_rows(dataset, "train")
    count = round(float(size) * len(rows)) if "." in size else int(size)
    if count >= len(rows):
        return rows
    chosen = set(random.Random(seed).sample(range(len(rows)), count))
    return [row for index, row in enumerate(rows) if index in chosen]


def subset_path(dataset, subset):
    return SUBSETS / dataset / f"{subset}.txt"


def spans_to_tags(length, spans):
    """Write (start, end, type) spans as IOB2, keeping longer spans and then earlier ones on overlap."""
    tags = ["O"] * length
    for start, end, kind in sorted(set(spans), key=lambda span: (span[0] - span[1], span[0])):
        if end <= start or any(tag != "O" for tag in tags[start:end]):
            continue
        tags[start] = f"B-{kind}"
        tags[start + 1:end] = [f"I-{kind}"] * (end - start - 1)
    return tags


def squeeze(text):
    return "".join(text.split())


def locate(tokens, text):
    """Find every token span whose characters equal the entity text, ignoring spaces and then case."""
    target = squeeze(text)
    if not target:
        return []
    for fold in (str, str.casefold):
        wanted = fold(target)
        pieces = [fold(squeeze(token)) for token in tokens]
        found = []
        for start in range(len(pieces)):
            joined = ""
            for end in range(start, len(pieces)):
                joined += pieces[end]
                if len(joined) >= len(wanted):
                    if joined == wanted:
                        found.append((start, end + 1))
                    break
        if found:
            return found
    return []


def entities_to_tags(tokens, entities, types):
    """Place extracted entity strings on the original tokens and count those that cannot be placed."""
    spans = []
    unknown_type = absent = 0
    for entity in entities:
        kind = entity["type"]
        if kind not in types:
            unknown_type += 1
            continue
        places = locate(tokens, entity["text"])
        if not places:
            absent += 1
        spans.extend((start, end, kind) for start, end in places)
    return spans_to_tags(len(tokens), spans), {"unknown_type": unknown_type, "absent_from_sentence": absent}


def char_spans_to_tags(tokens, spans):
    """Convert character spans over the space-joined tokens to IOB2 on those tokens."""
    starts, position = [], 0
    for token in tokens:
        starts.append(position)
        position += len(token) + 1
    token_spans = []
    for left, right, kind in spans:
        covered = [index for index, start in enumerate(starts)
                   if start < right and start + len(tokens[index]) > left]
        if covered:
            token_spans.append((covered[0], covered[-1] + 1, kind))
    return spans_to_tags(len(tokens), token_spans)


def write_predictions(dataset, method, rows, tags_by_id, seed=0):
    path = PREDS / dataset / method / f"{seed}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            record = {"sample_id": row["sample_id"], "tags": tags_by_id[row["sample_id"]]}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def score_rows(rows, tags_by_id):
    gold = {row["sample_id"]: row["tags"] for row in rows}
    return score_ner(gold, tags_by_id)


def score_any(rows, tags_by_id, target, schema):
    """Score in the target labels when the answer uses them, otherwise in the shared PER, LOC, ORG types."""
    if set(TYPE_DESCRIPTIONS[target]) == set(TYPE_DESCRIPTIONS[schema]):
        return {"label_space": "native", **score_rows(rows, tags_by_id)}
    gold = {row["sample_id"]: row["tags"] for row in rows}
    scores = score_transfer(gold, tags_by_id, target_dataset=target, prediction_schema=schema)
    return {"label_space": "shared PER, LOC, ORG", **scores}


def per_type_scores(rows, tags_by_id):
    """Score each gold or predicted entity type on its own."""
    gold = {row["sample_id"]: row["tags"] for row in rows}
    kinds = sorted({tag[2:] for tags in (*gold.values(), *tags_by_id.values()) for tag in tags if tag != "O"})
    keep = lambda tags, kind: [tag if tag[2:] == kind else "O" for tag in tags]
    return {
        kind: score_ner({key: keep(tags, kind) for key, tags in gold.items()},
                        {key: keep(tags, kind) for key, tags in tags_by_id.items()})
        for kind in kinds
    }


def save_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
