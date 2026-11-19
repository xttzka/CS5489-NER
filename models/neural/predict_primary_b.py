"""Extract entities with the supervised primary model fixed in a B manifest.

The manifest must be the full, unmodified frozen CoNLL selection. The chosen
checkpoint and its model/tokenizer assets are verified before forwarding raw
text to ``predict_sentence_b``. This program does not open any test split.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[2]
METHODS = {"M3": "bilstm", "M4": "bert", "Span": "span"}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def project_path(value, root):
    path = Path(value)
    result = (path if path.is_absolute() else root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Frozen artifact is outside the project directory: {value}")
    return result


def resolve_primary(manifest_path, root=PROJECT):
    """Validate the manifest and model files; never open its test input paths."""
    raw = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Frozen manifest must be a JSON object")
    signed = dict(raw)
    expected = signed.pop("manifest_sha256", None)
    canonical = json.dumps(signed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    if (raw.get("version") != 1 or raw.get("state") != "frozen" or raw.get("partial") is not False
            or raw.get("missing_requirements") != [] or expected != hashlib.sha256(canonical).hexdigest()):
        raise ValueError("Invalid, partial or modified frozen manifest")
    selection = raw.get("selection")
    if not isinstance(selection, dict) or (selection.get("dataset"), selection.get("split"), selection.get("scope")) != (
            "conll2003", "validation", "full"):
        raise ValueError("Manifest lacks the expected CoNLL full-validation model selection")
    method, seed = selection.get("primary_method"), selection.get("primary_seed")
    if method not in METHODS:
        raise ValueError(f"Selected method {method!r} is unsupported by this neural-only entry point")
    if type(seed) is not int or seed < 0:
        raise ValueError("Selected primary seed is missing or invalid")
    matches = [task for task in raw.get("tasks", [])
               if (task.get("group"), task.get("experiment"), task.get("condition"), task.get("method"),
                   task.get("train_source"), task.get("target"), task.get("scope"), task.get("seed")) ==
               ("domain", "baseline", "standard", method, "conll2003", "conll2003", "full", seed)]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen CoNLL task for {method} seed {seed}; found {len(matches)}")
    task = matches[0]
    run_dir = project_path(task["run_dir"], root)
    checkpoint = project_path(task["checkpoint"], root)
    if not checkpoint.is_relative_to(run_dir):
        raise ValueError("Selected checkpoint is outside its training run directory")
    record = raw.get("runs", {}).get(task["run_dir"])
    if not isinstance(record, dict) or project_path(record.get("checkpoint", ""), root) != checkpoint:
        raise ValueError("Selected task and frozen checkpoint inventory disagree")
    predictions = [row for row in selection.get("predictions", {}).get(method, []) if row.get("seed") == seed]
    if (len(predictions) != 1
            or project_path(predictions[0].get("path", ""), root) != run_dir / "validation_predictions.jsonl"
            or predictions[0].get("sha256") != record.get("validation_predictions_sha256")):
        raise ValueError("Selected validation evidence does not identify this training run")
    checks = {checkpoint: record["checkpoint_sha256"],
              run_dir / "run_config.json": record["run_config_sha256"],
              run_dir / "validation_predictions.jsonl": record["validation_predictions_sha256"],
              run_dir / "validation_scores.json": record["validation_scores_sha256"]}
    assets = record.get("assets")
    if not isinstance(assets, dict) or not assets:
        raise ValueError("Frozen model/tokenizer asset inventory is missing")
    for name, asset_hash in assets.items():
        asset = project_path(name, root)
        if not asset.is_relative_to(run_dir):
            raise ValueError(f"Model asset is outside its training run directory: {asset}")
        if asset in checks and checks[asset] != asset_hash:
            raise ValueError(f"Frozen hashes disagree for {asset}")
        checks[asset] = asset_hash
    for path, expected_hash in checks.items():
        if not path.is_file() or digest(path) != expected_hash:
            raise ValueError(f"Frozen model or validation artifact is missing or changed: {path}")
    frozen_code = raw.get("code_sha256", {})
    for name in ("models/neural/predict_neural.py", "models/neural/bert_baseline.py",
                 "models/neural/bilstm_crf.py", "models/neural/span_bert.py", "models/neural/token_alignment.py", "eval/score_ner.py"):
        file = project_path(name, root)
        if name not in frozen_code or not file.is_file() or digest(file) != frozen_code[name]:
            raise ValueError(f"Frozen inference code is missing or changed: {name}")
    return {"method": METHODS[method], "primary_method": method, "seed": seed,
            "run_dir": str(run_dir), "manifest_sha256": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "output/b_finish_v2/frozen.json",
                        help="Frozen selection; defaults to alignment-fix v2. Historic v1 requires its matching original code in an isolated copy.")
    parser.add_argument("--text", required=True, action="append", help="Original text; repeat for multiple inputs")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pos-python", type=Path, default=PROJECT / ".venv-b-pos/bin/python")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        chosen = resolve_primary(args.manifest)
        command = [sys.executable, "-B", "-m", "models.neural.predict_sentence_b",
                   "--method", chosen["method"], "--model", chosen["run_dir"],
                   "--device", args.device, "--batch-size", str(args.batch_size),
                   "--pos-python", str(args.pos_python)]
        for value in args.text:
            command.extend(("--text", value))
        print(f"Frozen primary: {chosen['primary_method']} seed {chosen['seed']} (device: {args.device})", file=sys.stderr)
        return subprocess.run(command, cwd=PROJECT, check=False).returncode
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
