"""Train the M2 linear-chain CRF and predict on the validation split."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from sklearn_crfsuite import CRF

from models.classic.features import sentence_features
from models.classic.m0_dictionary import DATA_DIR, PROJECT_DIR, read_sentences
from models.classic.m1_logistic import repair_iob2


DEFAULT_OUTPUT = PROJECT_DIR / "preds" / "conll2003" / "m2" / "0.jsonl"


def train_model(sentences, model_path, *, c1=0.1, c2=0.1):
    if not sentences:
        raise ValueError("training file contains no sentences")

    labels = []
    for row in sentences:
        if len(row["tokens"]) != len(row["tags"]):
            raise ValueError(f"{row['sample_id']}: token and tag counts differ")
        labels.append(row["tags"])

    # Keep each sentence intact so the CRF learns transitions within sentences.
    classifier = CRF(
        algorithm="lbfgs",
        c1=c1,
        c2=c2,
        max_iterations=100,
        all_possible_transitions=True,
        model_filename=str(model_path),
    )
    token_count = sum(map(len, labels))
    print(f"Training CRF on {len(sentences):,} sentences and {token_count:,} tokens", flush=True)
    classifier.fit((sentence_features(row) for row in sentences), labels)
    return classifier


def write_predictions(path, sentences, classifier):
    corrected_tokens = corrected_sentences = 0
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in sentences:
            raw_tags = classifier.predict_single(sentence_features(row))
            # Count and repair the same invalid transitions as M1 before scoring.
            tags, corrections = repair_iob2(raw_tags)
            prediction = {"sample_id": row["sample_id"], "tags": tags}
            target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            corrected_tokens += corrections
            corrected_sentences += int(corrections > 0)
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
        if not validation:
            raise ValueError("validation file contains no sentences")
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as staging:
            stage = Path(staging)
            classifier = train_model(training, stage / "model.crfsuite")
            predictions = stage / "predictions.jsonl"
            try:
                corrected_tokens, corrected_sentences = write_predictions(
                    predictions, validation, classifier
                )
            finally:
                classifier.tagger_.close()
            predictions.replace(output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")

    print(f"Wrote {len(validation):,} validation predictions to {output}")
    print(f"IOB2 repairs: {corrected_tokens:,} tokens across {corrected_sentences:,} sentences")


if __name__ == "__main__":
    main()
