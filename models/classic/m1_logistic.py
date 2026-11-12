"""Train the M1 token classifier and predict on the validation split."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from eval.score_ner import score_ner
from models.classic.features import sentence_features
from models.classic.m0_dictionary import DATA_DIR, PROJECT_DIR, read_sentences


DEFAULT_OUTPUT = PROJECT_DIR / "preds" / "conll2003" / "m1" / "0.jsonl"
C_VALUES = (0.1, 1.0, 10.0)
SOLVER = "lbfgs"
MAX_ITER = 2_000
TOL = 1e-5


def token_features(sentences):
    for row in sentences:
        yield from sentence_features(row)


def training_matrix(sentences):
    labels = []
    for row in sentences:
        if len(row["tokens"]) != len(row["tags"]):
            raise ValueError(f"{row['sample_id']}: token and tag counts differ")
        labels.extend(row["tags"])
    if not labels:
        raise ValueError("training file contains no tokens")

    # Fit the feature vocabulary on training text only.
    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(token_features(sentences))
    print(f"Training on {matrix.shape[0]:,} tokens and {matrix.shape[1]:,} feature columns", flush=True)
    return vectorizer, matrix, labels


def fit_classifier(matrix, labels, c):
    classifier = LogisticRegression(C=c, solver=SOLVER, max_iter=MAX_ITER, tol=TOL)
    with threadpool_limits(limits=2):
        classifier.fit(matrix, labels)
    print(f"M1 C={c:g} | training iterations: {classifier.n_iter_[0]}", flush=True)
    return classifier


def train_model(sentences, *, c=1.0):
    vectorizer, matrix, labels = training_matrix(sentences)
    return vectorizer, fit_classifier(matrix, labels, c)


def select_model(training, validation):
    """Choose C by exact validation F1 with a training-only feature vocabulary."""
    if not validation:
        raise ValueError("validation file contains no sentences")
    vectorizer, matrix, labels = training_matrix(training)
    validation_matrix = vectorizer.transform(token_features(validation))
    gold = {row["sample_id"]: row["tags"] for row in validation}
    scores = []
    best_model = None
    best_f1 = -1.0
    for c in C_VALUES:
        classifier = fit_classifier(matrix, labels, c)
        raw_tags = classifier.predict(validation_matrix).tolist()
        predictions = {sample_id: tags for sample_id, tags, _ in repaired_sentences(validation, raw_tags)}
        result = score_ner(gold, predictions)
        scores.append({"c": c, "iterations": int(classifier.n_iter_[0]), **result})
        print(f"M1 C={c:g} | validation F1={result['f1']:.4%}", flush=True)
        if result["f1"] > best_f1:
            best_model, best_f1 = classifier, result["f1"]
    return vectorizer, best_model, scores


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


def repaired_sentences(sentences, raw_tags):
    offset = 0
    for row in sentences:
        end = offset + len(row["tokens"])
        tags, corrections = repair_iob2(raw_tags[offset:end])
        yield row["sample_id"], tags, corrections
        offset = end


def write_predictions(path, sentences, vectorizer, classifier):
    if not sentences:
        raise ValueError("validation file contains no sentences")
    # Transform with the same columns learned during training.
    raw_tags = classifier.predict(vectorizer.transform(token_features(sentences))).tolist()

    path.parent.mkdir(parents=True, exist_ok=True)
    corrected_tokens = corrected_sentences = 0
    with TemporaryDirectory(dir=path.parent) as staging:
        temporary_file = Path(staging) / path.name
        with temporary_file.open("w", encoding="utf-8", newline="\n") as target:
            for sample_id, tags, corrections in repaired_sentences(sentences, raw_tags):
                prediction = {"sample_id": sample_id, "tags": tags}
                target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                corrected_tokens += corrections
                corrected_sentences += int(corrections > 0)
        temporary_file.replace(path)
    return corrected_tokens, corrected_sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.jsonl")
    parser.add_argument("--validation", type=Path, default=DATA_DIR / "validation.jsonl")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--c", type=float, help="Use a fixed C instead of selecting it on the validation split")
    args = parser.parse_args()

    output = args.output.resolve()
    if output in {args.train.resolve(), args.validation.resolve()}:
        parser.error("--output must differ from the training and validation files")
    try:
        training = list(read_sentences(args.train, "train"))
        validation = list(read_sentences(args.validation, "validation"))
        if args.c is None:
            vectorizer, classifier, _ = select_model(training, validation)
        else:
            vectorizer, classifier = train_model(training, c=args.c)
        corrected_tokens, corrected_sentences = write_predictions(
            output, validation, vectorizer, classifier
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")

    print(f"Wrote {len(validation):,} validation predictions with C={classifier.C:g} to {output}")
    print(f"IOB2 repairs: {corrected_tokens:,} tokens across {corrected_sentences:,} sentences")


if __name__ == "__main__":
    main()
