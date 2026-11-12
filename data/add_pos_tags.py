"""Add spaCy POS tags to copies of the processed NER datasets."""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import spacy
from spacy.tokens import Doc


DATA_DIR = Path(__file__).resolve().parent
INPUT_ROOT = DATA_DIR / "processed"
OUTPUT_ROOT = DATA_DIR / "processed_pos"
MODEL = "en_core_web_sm"
MODEL_VERSION = "3.8.0"
STANDARD_SPLITS = ("train", "validation", "test")
DATASETS = {
    "conll2003": STANDARD_SPLITS,
    "conllpp": ("test",),
    "wikigold": ("test",),
    "btc": ("test",),
    "wnut17": STANDARD_SPLITS,
    "mit_restaurant": STANDARD_SPLITS,
}


def tag_texts(nlp, texts):
    if not isinstance(texts, list) or not texts or not all(
        isinstance(text, str) and text.strip() for text in texts
    ):
        raise ValueError("expected a JSON list of nonempty input strings")
    for doc in nlp.pipe(texts):
        words = [token for token in doc if not token.is_space]
        if not words or not all(token.tag_ for token in words):
            raise ValueError("each input needs at least one token and a POS tag for each token")
        yield {
            "text": doc.text,
            "tokens": [token.text for token in words],
            "pos_tags": [token.tag_ for token in words],
            "offsets": [[token.idx, token.idx + len(token.text)] for token in words],
        }


def read_docs(nlp, source, dataset, split):
    with source.open(encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, 1):
            row = json.loads(line)
            if row["dataset"] != dataset or row["split"] != split:
                raise ValueError(f"{source}:{line_number}: expected {dataset}/{split}")
            tokens = row["tokens"]
            if not tokens or len(tokens) != len(row["tags"]):
                raise ValueError(f"{row['sample_id']}: tokens and entity tags must align")
            # Supplying a Doc lets the tagger use the dataset's token boundaries.
            yield Doc(nlp.vocab, words=tokens), row


def tag_split(nlp, dataset, split):
    source = INPUT_ROOT / dataset / f"{split}.jsonl"
    output = OUTPUT_ROOT / dataset / source.name
    output.parent.mkdir(parents=True, exist_ok=True)
    sentence_count = token_count = 0
    with TemporaryDirectory(dir=output.parent) as staging:
        staged = Path(staging) / output.name
        with staged.open("w", encoding="utf-8", newline="\n") as target:
            docs = read_docs(nlp, source, dataset, split)
            for doc, row in nlp.pipe(docs, as_tuples=True, batch_size=128):
                if [token.text for token in doc] != row["tokens"]:
                    raise ValueError(f"{row['sample_id']}: the tagger changed the original tokens")
                pos_tags = [token.tag_ for token in doc]
                if not all(pos_tags):
                    raise ValueError(f"{row['sample_id']}: at least one POS tag is missing")
                row["pos_tags"] = pos_tags
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                sentence_count += 1
                token_count += len(pos_tags)
        if not sentence_count:
            raise ValueError(f"{source}: no sentences to tag")
        staged.replace(output)

    print(f"{dataset}/{split}: {sentence_count:,} sentences, {token_count:,} POS tags", flush=True)
    return sentence_count, token_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dataset", choices=DATASETS, help="Tag one dataset instead of all six")
    mode.add_argument("--text-json", action="store_true", help="Tag raw strings from a UTF-8 JSON list on stdin")
    args = parser.parse_args()

    try:
        nlp = spacy.load(MODEL, enable=["tok2vec", "tagger"])
        if nlp.meta["version"] != MODEL_VERSION:
            raise ValueError(f"expected {MODEL} {MODEL_VERSION}, found {nlp.meta['version']}")
        if args.text_json:
            texts = json.loads(sys.stdin.buffer.read().decode("utf-8"))
            result = json.dumps(list(tag_texts(nlp, texts)), ensure_ascii=False)
            sys.stdout.buffer.write((result + "\n").encode("utf-8"))
            return
        print(f"spaCy {spacy.__version__}, {MODEL} {MODEL_VERSION}", flush=True)

        sentences = tokens = 0
        for dataset in [args.dataset] if args.dataset else DATASETS:
            for split in DATASETS[dataset]:
                split_sentences, split_tokens = tag_split(nlp, dataset, split)
                sentences += split_sentences
                tokens += split_tokens
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")

    print(f"Total: {sentences:,} sentences, {tokens:,} POS tags. Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
