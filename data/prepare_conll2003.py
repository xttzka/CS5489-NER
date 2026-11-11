"""Convert the CoNLL-2003 archive used by eriktks/conll2003 to JSONL."""

import argparse
import json
from io import TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


DATASET = "conll2003"
DATA_DIR = Path(__file__).resolve().parent
SPLITS = {
    # Split name: (archive member, expected sentences, expected documents)
    "train": ("train.txt", 14_041, 946),
    "validation": ("valid.txt", 3_250, 216),
    "test": ("test.txt", 3_453, 231),
}
NER_TAGS = {
    "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
    "B-LOC", "I-LOC", "B-MISC", "I-MISC",
}


def read_sentences(lines, filename):
    """Yield each sentence's document number and four-column token rows."""
    document_number = 0
    sentence = []

    for line_number, line in enumerate(lines, start=1):
        columns = line.split()
        is_document_start = bool(columns) and columns[0] == "-DOCSTART-"
        if not columns or is_document_start:
            if sentence:
                yield document_number, sentence
                sentence = []
            if is_document_start:
                document_number += 1
            continue

        location = f"{filename}:{line_number}"
        if len(columns) != 4 or document_number == 0:
            raise ValueError(f"{location}: expected four columns within a document")

        tag = columns[3]
        if tag not in NER_TAGS:
            raise ValueError(f"{location}: unknown NER tag {tag!r}")
        if tag.startswith("I-"):
            previous_tag = sentence[-1][3] if sentence else "O"
            if previous_tag not in {tag, "B-" + tag[2:]}:
                raise ValueError(f"{location}: {tag} cannot follow {previous_tag} in IOB2")
        sentence.append(columns)

    if sentence:
        yield document_number, sentence


def convert_split(archive, split, output_file):
    filename, expected_sentences, expected_documents = SPLITS[split]
    document_numbers = set()
    sentence_count = 0
    with TextIOWrapper(archive.open(filename), encoding="utf-8") as source:
        with output_file.open("w", encoding="utf-8", newline="\n") as target:
            for document_number, sentence in read_sentences(source, filename):
                tokens, pos_tags, chunk_tags, tags = zip(*sentence)
                record = {
                    "dataset": DATASET,
                    "split": split,
                    "sample_id": f"{DATASET}:{split}:{sentence_count}",
                    "doc_id": f"{DATASET}:{split}:{document_number}",
                    "tokens": tokens,
                    "tags": tags,
                    "pos_tags": pos_tags,
                    "chunk_tags": chunk_tags,
                }
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                document_numbers.add(document_number)
                sentence_count += 1

    document_count = len(document_numbers)
    if (sentence_count, document_count) != (expected_sentences, expected_documents):
        raise ValueError(
            f"{filename}: got {sentence_count} sentences and {document_count} documents, "
            f"expected {expected_sentences} sentences and {expected_documents} documents"
        )
    return sentence_count, document_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DATA_DIR / "raw" / DATASET / "conll2003.zip")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "processed" / DATASET)
    args = parser.parse_args()

    with ZipFile(args.archive) as archive:
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output_dir.parent) as temporary_directory:
            staging_dir = Path(temporary_directory)
            counts = {
                split: convert_split(archive, split, staging_dir / f"{split}.jsonl")
                for split in SPLITS
            }

            # Publish only after all three splits pass validation.
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for split, (sentences, documents) in counts.items():
                filename = f"{split}.jsonl"
                output_file = args.output_dir / filename
                (staging_dir / filename).replace(output_file)
                print(f"{split}: {sentences:,} sentences, {documents:,} documents -> {output_file}")


if __name__ == "__main__":
    main()
