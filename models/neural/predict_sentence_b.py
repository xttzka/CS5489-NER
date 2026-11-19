"""Extract entities and character offsets from raw strings using a local neural checkpoint.

Offsets are zero-based Python Unicode code-point indices with exclusive ends.
The tokenizer/model matches the project's existing raw-text prediction path.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from data.prepare import pos_python
from eval.score_ner import bio_to_spans


PROJECT = Path(__file__).resolve().parents[2]


def run_child(command, *, data=None, environment=None):
    result = subprocess.run(
        command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=PROJECT, env=environment, check=False,
    )
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
        sys.stderr.buffer.flush()
    if result.returncode:
        raise RuntimeError(f"{command[3]} failed with exit status {result.returncode}")
    return result.stdout


def tokenize_texts(texts, interpreter):
    data = json.dumps(texts, ensure_ascii=False).encode("utf-8")
    raw = run_child([str(interpreter), "-B", "-m", "data.add_pos_tags", "--text-json"], data=data)
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise ValueError("POS tokenizer returned an unexpected number of records")
    result = []
    for index, (text, row) in enumerate(zip(texts, rows, strict=True)):
        tokens, offsets = row["tokens"], row["offsets"]
        if row["text"] != text or not tokens or len(tokens) != len(offsets):
            raise ValueError(f"input {index}: tokenizer returned inconsistent text or offsets")
        previous_end = 0
        for token, offset in zip(tokens, offsets, strict=True):
            if not isinstance(offset, list) or len(offset) != 2:
                raise ValueError(f"input {index}: invalid token offset")
            start, end = offset
            if (not isinstance(start, int) or not isinstance(end, int) or
                    start < previous_end or end <= start or end > len(text) or text[start:end] != token):
                raise ValueError(f"input {index}: token offset does not match original text")
            previous_end = end
        result.append({"sample_id": f"input:{index}", "text": text, "tokens": tokens, "offsets": offsets})
    return result


def extract(rows, answers):
    if set(answers) != {row["sample_id"] for row in rows}:
        raise ValueError("prediction IDs do not match the input IDs")
    output = []
    for row in rows:
        tags = answers[row["sample_id"]]
        if len(tags) != len(row["tokens"]):
            raise ValueError(f'{row["sample_id"]}: predicted tag count differs from token count')
        entities = []
        for start, end, kind in sorted(bio_to_spans(tags)):
            left, right = row["offsets"][start][0], row["offsets"][end - 1][1]
            entities.append({"text": row["text"][left:right], "type": kind, "start": left, "end": right})
        output.append({"text": row["text"], "entities": entities})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("bert", "bilstm", "span"))
    parser.add_argument("--model", required=True, help="Local training output directory or checkpoint")
    parser.add_argument("--text", action="append", required=True, help="Original text; repeat for multiple inputs")
    parser.add_argument("--device", required=True, choices=("cpu", "mps"), help="Explicit inference device")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pos-python", type=Path, help="Python with the project's pinned POS model")
    parser.add_argument("--tokenizer", help="Local Span tokenizer directory if not beside checkpoint")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        rows = tokenize_texts(args.text, pos_python(args.pos_python))
        with TemporaryDirectory(prefix="ner_sentence_b_") as directory:
            work = Path(directory)
            input_path, output_path = work / "input.jsonl", work / "prediction.jsonl"
            with input_path.open("x", encoding="utf-8") as target:
                for row in rows:
                    # No gold labels, POS tags or offsets are passed into the NER model.
                    target.write(json.dumps({"sample_id": row["sample_id"], "tokens": row["tokens"]}, ensure_ascii=False) + "\n")
            command = [sys.executable, "-B", "-m", "models.neural.predict_neural", "--method", args.method,
                       "--model", args.model, "--input", str(input_path), "--output", str(output_path),
                       "--device", args.device, "--batch-size", str(args.batch_size)]
            if args.tokenizer:
                command.extend(["--tokenizer", args.tokenizer])
            environment = os.environ.copy()
            environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
            progress = run_child(command, environment=environment)
            if progress:
                sys.stderr.buffer.write(progress)
            answers = {}
            with output_path.open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if row["sample_id"] in answers:
                        raise ValueError(f'duplicate prediction ID: {row["sample_id"]}')
                    answers[row["sample_id"]] = row["tags"]
            predictions = extract(rows, answers)
        output = {"method": args.method, "model": args.model, "device": args.device,
                  "offset_convention": "zero-based Unicode code points, end exclusive",
                  "predictions": predictions}
        sys.stdout.buffer.write((json.dumps(output, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
