"""Download or convert the five additional datasets in the NER project."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen


DATA_DIR = Path(__file__).resolve().parent
GITHUB_RAW = "https://raw.githubusercontent.com"
WNUT_ROOT = f"{GITHUB_RAW}/leondz/emerging_entities_17/master"
MIT_ROOT = "https://sls.csail.mit.edu/downloads/restaurant"
SOURCES = {
    "wnut17": {
        "wnut17train.conll": (
            f"{WNUT_ROOT}/wnut17train.conll",
            "731820e13f71af324c6b55a1575ec2ce59fbaa2a0806f8f0400b98d56cd6a7a5",
        ),
        "emerging.dev.conll": (
            f"{WNUT_ROOT}/emerging.dev.conll",
            "e053b752b8155113ca66e7db08af5f2f5f43ab993694e4774f1ec5bad42312d8",
        ),
        "emerging.test.annotated": (
            f"{WNUT_ROOT}/emerging.test.annotated",
            "2aa79b764e56ec9264a1b30fdd9b70195bd00ff400b62edd8f399d5f13c178f0",
        ),
    },
    "mit_restaurant": {
        "restauranttrain.bio": (
            f"{MIT_ROOT}/restauranttrain.bio",
            "f51794247426c975e7701011d4dba668ac6b881358f162e2c2c10709f2d17b9b",
        ),
        "restauranttest.bio": (
            f"{MIT_ROOT}/restauranttest.bio",
            "09809dd126a3781c4205d06e721752054e51a10b1c98f3495ef17cd723f25ce2",
        ),
    },
    "conllpp": {
        "conllpp.txt": (
            f"{GITHUB_RAW}/ShuhengL/acl2023_conllpp/main/dataset/conllpp.txt",
            "bd08b7a49358e54f3f0a41e8431880ce85709e38b08a178f7f5e698d2e98353b",
        ),
    },
    "wikigold": {
        "wikigold.conll.txt": (
            f"{GITHUB_RAW}/juand-r/entity-recognition-datasets/master/"
            "data/wikigold/CONLL-format/data/wikigold.conll.txt",
            "c797a64d0cf73ed058363f77671486bcfd413e70fda1726ad4a6ba624455225f",
        ),
    },
    "btc": {
        "f.json": (
            f"{GITHUB_RAW}/GateNLP/broad_twitter_corpus/master/f.json",
            "f2d2e05a02a3ef841e8a91abacf4953d0a196ced655cfe2f7683cf7bddbfd1bc",
        ),
    },
}

COLUMN_SPLITS = {
    "wnut17": {
        "train": "wnut17train.conll",
        "validation": "emerging.dev.conll",
        "test": "emerging.test.annotated",
    },
    "mit_restaurant": {"train": "restauranttrain.bio", "test": "restauranttest.bio"},
    "conllpp": {"test": "conllpp.txt"},
    "wikigold": {"test": "wikigold.conll.txt"},
}
ENTITY_TYPES = {
    "wnut17": {"person", "location", "corporation", "group", "product", "creative-work"},
    "mit_restaurant": {
        "Amenity", "Cuisine", "Dish", "Hours", "Location", "Price", "Rating", "Restaurant_Name",
    },
    "conllpp": {"PER", "LOC", "ORG", "MISC"},
    "wikigold": {"PER", "LOC", "ORG", "MISC"},
    "btc": {"PER", "LOC", "ORG"},
}
# Sentences or queries or tweets, then tokens and entity spans after conversion.
EXPECTED = {
    "wnut17": {
        "train": (3394, 62730, 1975),
        "validation": (1009, 15733, 836),
        "test": (1287, 23394, 1079),
    },
    "mit_restaurant": {
        "train": (6894, 64465, 13708),
        "validation": (766, 6060, 1655),
        "test": (1521, 14256, 3151),
    },
    "conllpp": {"test": (1840, 46456, 4007)},
    "wikigold": {"test": (1696, 39007, 3558)},
    "btc": {"test": (2000, 35500, 4376)},
}
EXPECTED_DOCUMENTS = {"conllpp": 131, "wikigold": 145, "btc": 2000}
IOB1_DATASETS = {"conllpp", "wikigold"}


def check_digest(content, expected_digest, path):
    if sha256(content).hexdigest() != expected_digest:
        raise ValueError(f"{path}: SHA-256 does not match the verified source")


def prepare_sources(dataset, raw_dir, download):
    for filename, (url, digest) in SOURCES[dataset].items():
        path = raw_dir / filename
        if path.exists():
            check_digest(path.read_bytes(), digest, path)
            continue
        if not download:
            raise FileNotFoundError(f"{path}: run with --download first")

        print(f"Downloading {url}", flush=True)
        with urlopen(url, timeout=45) as response:
            content = response.read()
        check_digest(content, digest, path)
        raw_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=raw_dir) as staging:
            temporary_file = Path(staging) / filename
            temporary_file.write_bytes(content)
            temporary_file.replace(path)


def read_columns(path, dataset):
    """Read sentences as (document number, tokens, original NER tags)."""
    separator = "\t" if dataset in {"wnut17", "mit_restaurant"} else None
    expected_columns = 4 if dataset == "conllpp" else 2
    # CoNLL++ starts with a marker. WikiGold starts with the first word.
    document = {"conllpp": 0, "wikigold": 1}.get(dataset)
    tokens, tags = [], []
    with path.open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            columns = line.rstrip("\r\n").split(separator) if line.strip() else []
            is_marker = bool(columns) and columns[0] == "-DOCSTART-"
            if not columns or is_marker:
                if tokens:
                    yield document, tokens, tags
                    tokens, tags = [], []
                if is_marker:
                    if document is None:
                        raise ValueError(f"{path}:{line_number}: unexpected document marker")
                    document += 1
                continue

            if len(columns) != expected_columns or document == 0:
                raise ValueError(f"{path}:{line_number}: invalid source record")
            if dataset == "mit_restaurant":
                tag, token = columns
            else:
                token, tag = columns[0], columns[-1]
            if dataset == "wnut17":
                token = token.replace("\ufeff", "")
            tokens.append(token)
            tags.append(tag.strip())

    if tokens:
        yield document, tokens, tags


def read_btc(path):
    """Keep one record per tweet and use the author's token and NER arrays."""
    with path.open(encoding="utf-8") as source:
        for document, line in enumerate(source, start=1):
            tweet = json.loads(line)
            tokens, tags = [], []
            for token, tag in zip(tweet["tokens"], tweet["entities"], strict=True):
                # The release contains three whitespace-only tokens tagged O.
                if not token.strip() and tag == "O":
                    continue
                tokens.append(token)
                tags.append(tag)
            yield document, tokens, tags


def load_splits(dataset, raw_dir):
    if dataset == "btc":
        return {"test": read_btc(raw_dir / "f.json")}
    splits = {
        split: read_columns(raw_dir / filename, dataset)
        for split, filename in COLUMN_SPLITS[dataset].items()
    }
    if dataset == "mit_restaurant":
        # Use the last contiguous 10% of the official training file as dev.
        original_train = list(splits["train"])
        boundary = len(original_train) * 9 // 10
        splits["train"] = original_train[:boundary]
        splits["validation"] = original_train[boundary:]
    return splits


def normalize_tags(tags, dataset, sample_id):
    """Normalize boundary prefixes while keeping the source entity types."""
    normalized = []
    previous = "O"
    for tag in tags:
        if tag != "O":
            prefix, _, entity_type = tag.partition("-")
            if prefix not in {"B", "I"} or entity_type not in ENTITY_TYPES[dataset]:
                raise ValueError(f"{sample_id}: unknown NER tag {tag!r}")
            if prefix == "I" and previous not in {tag, f"B-{entity_type}"}:
                if dataset not in IOB1_DATASETS:
                    raise ValueError(f"{sample_id}: {tag} cannot follow {previous} in IOB2")
                tag = f"B-{entity_type}"
        normalized.append(tag)
        previous = tag
    return normalized


def write_split(path, dataset, split, sentences):
    sentence_count = token_count = entity_count = 0
    documents = set()
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for document, tokens, original_tags in sentences:
            sample_id = f"{dataset}:{split}:{sentence_count}"
            if len(tokens) != len(original_tags) or not tokens or any(not t.strip() for t in tokens):
                raise ValueError(f"{sample_id}: invalid token or tag alignment")
            tags = normalize_tags(original_tags, dataset, sample_id)
            record = {
                "dataset": dataset,
                "split": split,
                "sample_id": sample_id,
                "doc_id": None if document is None else f"{dataset}:{split}:{document}",
                "tokens": tokens,
                "tags": tags,
                "pos_tags": None,
                "chunk_tags": None,
            }
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            if document is not None:
                documents.add(document)
            sentence_count += 1
            token_count += len(tokens)
            entity_count += sum(tag.startswith("B-") for tag in tags)

    actual = (sentence_count, token_count, entity_count)
    expected = EXPECTED[dataset][split]
    expected_docs = EXPECTED_DOCUMENTS.get(dataset, 0)
    if actual != expected or len(documents) != expected_docs:
        raise ValueError(
            f"{dataset}/{split}: got {actual} and {len(documents)} docs, "
            f"expected {expected} and {expected_docs} docs"
        )
    return actual


def convert_dataset(dataset, raw_dir, output_root):
    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_root) as staging:
        staging_dir = Path(staging)
        counts = {}
        for split, sentences in load_splits(dataset, raw_dir).items():
            counts[split] = write_split(staging_dir / f"{split}.jsonl", dataset, split, sentences)

        output_dir = output_root / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        for split, (records, tokens, entities) in counts.items():
            filename = f"{split}.jsonl"
            (staging_dir / filename).replace(output_dir / filename)
            print(f"{dataset}/{split}: {records:,} records, {tokens:,} tokens, {entities:,} entities")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, help="Process only this dataset")
    parser.add_argument("--download", action="store_true", help="Download missing sources and exit")
    args = parser.parse_args()

    for dataset in [args.dataset] if args.dataset else SOURCES:
        raw_dir = DATA_DIR / "raw" / dataset
        prepare_sources(dataset, raw_dir, args.download)
        if not args.download:
            convert_dataset(dataset, raw_dir, DATA_DIR / "processed")


if __name__ == "__main__":
    main()
