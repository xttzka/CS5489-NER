"""Train the M1 token classifier and predict on the validation split."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from models.classic.features import sentence_features
from models.classic.m0_dictionary import DATA_DIR, PROJECT_DIR, read_sentences


DEFAULT_OUTPUT = PROJECT_DIR / "preds" / "conll2003" / "m1" / "0.jsonl"


def train_model(sentences):
    labels = []
    for row in sentences:
        if len(row["tokens"]) != len(row["tags"]):
            raise ValueError(f"{row['sample_id']}: token and tag counts differ")
        labels.extend(row["tags"])
    if not labels:
        raise ValueError("training file contains no tokens")

    # Fit the feature vocabulary on training text only.
    vectorizer = DictVectorizer(sparse=True)
    inputs = (feature for row in sentences for feature in sentence_features(row))
    matrix = vectorizer.fit_transform(inputs)
    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        tol=1e-3,
    )
    print(f"Training on {matrix.shape[0]:,} tokens and {matrix.shape[1]:,} feature columns", flush=True)
    with threadpool_limits(limits=2):
        classifier.fit(matrix, labels)
    print(f"Training iterations: {classifier.n_iter_[0]}", flush=True)
    return vectorizer, classifier


def repair_iob2(tags):
    """Start an entity where an independent I-tag has no matching predecessor."""
    repaired = []
    corrections = 0
    previous = "O"
    for tag in tags:
        if tag.startswith("I-") and previous not in {tag, f"B-{tag[2:]}"}:
            tag = f"B-{tag[2:]}"
            corrections += 1
        repaired.append(tag)
        previous = tag
    return repaired, corrections


def write_predictions(path, sentences, vectorizer, classifier):
    if not sentences:
        raise ValueError("validation file contains no sentences")
    inputs = (feature for row in sentences for feature in sentence_features(row))
    # Transform with the same columns learned during training.
    raw_tags = classifier.predict(vectorizer.transform(inputs)).tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    corrected_tokens = corrected_sentences = 0
    offset = 0
    with TemporaryDirectory(dir=path.parent) as staging:
        temporary_file = Path(staging) / path.name
        with temporary_file.open("w", encoding="utf-8", newline="\n") as target:
            for row in sentences:
                end = offset + len(row["tokens"])
                tags, corrections = repair_iob2(raw_tags[offset:end])
                prediction = {"sample_id": row["sample_id"], "tags": tags}
                target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                corrected_tokens += corrections
                corrected_sentences += int(corrections > 0)
                offset = end
        temporary_file.replace(path)
    return corrected_tokens, corrected_sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    parser.add_argument("--validation", type=Path, default=DATA_DIR / "validation.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    if output in {args.train.resolve(), args.validation.resolve()}:
        parser.error("--output must differ from the training and validation files")
    try:
        training = list(read_sentences(args.train, "train"))
        validation = list(read_sentences(args.validation, "validation"))
        vectorizer, classifier = train_model(training)
        corrected_tokens, corrected_sentences = write_predictions(
            output, validation, vectorizer, classifier
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")

    print(f"Wrote {len(validation):,} validation predictions to {output}")
    print(f"IOB2 repairs: {corrected_tokens:,} tokens across {corrected_sentences:,} sentences")


if __name__ == "__main__":
    main()
