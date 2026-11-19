"""Extract typed entity phrases from raw sentences with a classic model or the fine-tuned BERT tagger."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from data.prepare import pos_python
from eval.score_ner import bio_to_spans, read_tags_by_id
from models.classic import m0_dictionary, m1_logistic, m2_crf, run_baselines


PROJECT = m0_dictionary.PROJECT_DIR
DATASETS = ("conll2003", "wnut17", "mit_restaurant")
BERT_CHECKPOINT = PROJECT / "output" / "bert_main"
BERT_TRAINING = (
    "python -B -m models.neural.bert_baseline --dataset conll2003 --train-size 1.0 --dev-size 0 "
    "--epochs 3 --batch-size 16 --learning-rate 2e-5 --output-dir output/bert_main"
)


def predict_with_bert(texts, checkpoint, device, python):
    """Run the saved BERT tagger on raw sentences and return its entity phrases."""
    if not checkpoint.is_dir():
        raise ValueError(f"no trained BERT model at {checkpoint}. Train it first with: {BERT_TRAINING}")
    command = [sys.executable, "-B", "-m", "models.neural.predict_sentence", "--method", "bert",
               "--model", str(checkpoint), "--device", device, "--pos-python", str(python)]
    for text in texts:
        command += ["--text", text]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=PROJECT, check=False)
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "BERT prediction failed")
    return json.loads(result.stdout)["predictions"]


def tag_inputs(texts, python):
    result = subprocess.run(
        [str(python), "-B", "-m", "data.add_pos_tags", "--text-json"],
        input=json.dumps(texts, ensure_ascii=False).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT,
        check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.decode("utf-8", errors="replace").strip() or "POS tagging failed")
    rows = json.loads(result.stdout)
    for index, row in enumerate(rows):
        row["sample_id"] = f"input:{index}"
        row["chunk_tags"] = None
    return rows


def predict_answers(model, dataset, rows, work):
    train_path, training = run_baselines.load_rows(dataset, "train", shared_pos=True)
    sentences = list(training.values())
    path = work / "answers.jsonl"
    if model == "m0":
        dictionary, _ = m0_dictionary.build_dictionary(sentences)
        m0_dictionary.write_predictions(path, rows, dictionary)
    elif model == "m1":
        selection = run_baselines.read_m1_selection(dataset, train_path)
        vectorizer, classifier = m1_logistic.train_model(sentences, c=selection["c"])
        m1_logistic.write_predictions(path, rows, vectorizer, classifier)
    else:
        features = {"pos_model": run_baselines.POS_MODEL, "use_chunk_tags": False, "context_window": 1}
        parameters = run_baselines.m2_selection(dataset, train_path, features)["parameters"]
        classifier = m2_crf.train_model(sentences, work / "model.crfsuite", **parameters)
        try:
            m2_crf.write_predictions(path, rows, classifier)
        finally:
            classifier.tagger_.close()
    return read_tags_by_id(path)


def extract_phrases(rows, answers):
    for row in rows:
        entities = []
        for start, end, kind in sorted(bio_to_spans(answers[row["sample_id"]])):
            left, right = row["offsets"][start][0], row["offsets"][end - 1][1]
            entities.append({"text": row["text"][left:right], "type": kind, "start": left, "end": right})
        yield {"text": row["text"], "entities": entities}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str.lower, choices=("m0", "m1", "m2", "m4"), default="m2",
                        help="m0 dictionary, m1 logistic regression, m2 CRF, m4 fine-tuned BERT")
    parser.add_argument("--dataset", choices=DATASETS, default="conll2003", help="Training source of m0 to m2")
    parser.add_argument("--checkpoint", type=Path, default=BERT_CHECKPOINT,
                        help="Training output directory of the BERT tagger used by m4")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu", help="Device for m4")
    parser.add_argument("--text", action="append", required=True, help="Repeat for multiple inputs in the same fit")
    parser.add_argument("--pos-python", type=Path, help="Interpreter with the pinned spaCy model")
    args = parser.parse_args()
    try:
        if args.model == "m4":
            predictions = predict_with_bert(args.text, args.checkpoint.resolve(), args.device,
                                            pos_python(args.pos_python))
            output = {"model": args.model, "checkpoint": str(args.checkpoint), "predictions": predictions}
            sys.stdout.buffer.write((json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
            return
        rows = tag_inputs(args.text, pos_python(args.pos_python))
        with TemporaryDirectory(dir=run_baselines.RESULTS.parent) as directory, redirect_stdout(sys.stderr):
            answers = predict_answers(args.model, args.dataset, rows, Path(directory))
        output = {"model": args.model, "training_dataset": args.dataset, "predictions": list(extract_phrases(rows, answers))}
        sys.stdout.buffer.write((json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
