"""Dev-only, MPS-exclusive hyperparameter comparison for role B.

Each domain/model compares the existing seed-0 baseline against ONE joint
alternative fixed below. Epoch is selected by full-validation F1 at every
epoch. Seed-0 selects the configuration; seeds 1 and 2 then repeat it. The
joint candidates do not identify individual hyperparameter effects. No test
dataset is opened. Learning-curve and distillation runs retain their already
fixed exploratory schedules and are reported separately from these models.

The three-type transfer runs use the native source's selected BERT learning
rate and batch size. For merged sources, select between both fixed candidates
using only concatenated mapped source validation data. All writes preserve
previous artifacts; an incomplete attempt is retried in a new directory.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path

import torch

from models.neural.collect_b_results import inspect_run, queue_logs

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("conll2003", "wnut17", "mit_restaurant")
MODELS = ("bilstm", "bert", "span")
SPECS = {
    "bilstm": {
        "module": "models.neural.bilstm_crf",
        "base": {"epochs": 5, "batch_size": 16, "learning_rate": 0.001,
                 "hidden_size": 128, "dropout": 0.3},
        "alternate": {"epochs": 5, "batch_size": 16, "learning_rate": 0.0005,
                      "hidden_size": 96, "dropout": 0.5},
    },
    "bert": {
        "module": "models.neural.bert_extended",
        "base": {"epochs": 3, "batch_size": 8, "learning_rate": 3e-5},
        "alternate": {"epochs": 3, "batch_size": 16, "learning_rate": 2e-5},
    },
    "span": {
        "module": "models.neural.span_bert",
        "base": {"epochs": 3, "batch_size": 4, "learning_rate": 3e-5},
        "alternate": {"epochs": 3, "batch_size": 2, "learning_rate": 2e-5},
    },
}
QUEUE = ROOT / "output" / "b_training_queue"
OUT = ROOT / "output" / "b_tuning"


def config_matches(config, expected, seed, dataset=None, source=None, eval_every=1):
    for key, value in {**expected, "seed": seed, "dev_size": 0}.items():
        if config.get(key) != value:
            return False
    if config.get("train_size") != "1.0":
        return False
    if dataset and (config.get("dataset") != dataset or config.get("task", "native") != "native"):
        return False
    if source and (config.get("source") != source or config.get("task") != "transfer"):
        return False
    if config.get("device_used", config.get("device")) != "mps":
        return False
    if config.get("eval_every", 1) != eval_every:
        return False
    return True


def completed(base, spec, seed, dataset=None, source=None, eval_every=1, final_epoch=None):
    """Find first complete matching attempt using full validation and its log."""
    candidates = ([base] if base.exists() else []) + sorted(base.parent.glob(base.name + "__attempt_*"))
    known = queue_logs(ROOT)
    for candidate in candidates:
        info, _, _ = inspect_run(ROOT, candidate, known)
        if info["status"] == "complete" and config_matches(info["config"], spec, seed, dataset, source, eval_every):
            if final_epoch is not None:
                try:
                    check_final_epoch_only(info, final_epoch)
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
            return info
    return None


def next_path(base):
    if not base.exists() and not base.with_suffix(".log").exists():
        return base
    index = 2
    while True:
        path = base.with_name(base.name + f"__attempt_{index}")
        if not path.exists() and not path.with_suffix(".log").exists():
            return path
        index += 1


def command(model, dataset, spec, seed, path, source=None, eval_every=1):
    args = [sys.executable, "-u", "-B", "-m", SPECS[model]["module"]]
    if source is not None:
        if model != "bert":
            raise ValueError("only BERT supports transfer")
        args += ["--task", "transfer", "--source", source, "--device", "mps", "--local-files-only"]
    elif model == "bert":
        args += ["--task", "native", "--dataset", dataset, "--device", "mps", "--local-files-only"]
    else:
        args += ["--dataset", dataset]
        if model == "span":
            args += ["--max-span-length", "14", "--negative-ratio", "10"]
    args += ["--train-size", "1.0", "--dev-size", "0", "--seed", str(seed)]
    if model == "bert":
        args += ["--eval-every", str(eval_every)]
    for key, value in spec.items():
        args += ["--" + key.replace("_", "-"), str(value)]
    return args + ["--output-dir", str(path)]


def base_path(model, dataset, seed):
    return ROOT / "output" / f"{model}_{dataset}_full_seed{seed}"


def tuned_path(model, dataset, seed):
    return ROOT / "output" / f"{model}_tuning_{dataset}_alternate_seed{seed}"


def transfer_base(source):
    return ROOT / "output" / f"bert_transfer_{source}_seed0"


def transfer_tuned(source):
    return ROOT / "output" / f"bert_tuning_transfer_{source}_alternate_seed0"


def transfer_fixed(source, choice, epoch):
    return ROOT / "output" / f"bert_tuning_transfer_{source}_{choice}_finalepoch{epoch}_seed0"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(info):
    run_dir = Path(info["path"])
    hashes = {}
    for name in ("run_config.json", "validation_predictions.jsonl", "validation_scores.json"):
        hashes[name] = file_sha256(run_dir / name)
    return {"run_dir": info["path"], "validation_f1": info["scores"]["f1"], "sha256": hashes,
            "validation_predictions": info["prediction"], "checkpoint": info["checkpoint"],
            "completion_log": info["completion_log"],
            "completion_log_sha256": file_sha256(info["completion_log"])}


def native_best_epoch(info):
    """Reconstruct best epoch from complete native seed-0 dev evidence.

    New BERT training saves structured epoch metrics. The historical baseline
    instead prints the epoch heading followed by the multi-line JSON score.
    Validate all epochs and saved best F1; earliest epoch wins exact ties.
    """
    run_dir = Path(info["path"])
    metrics = run_dir / "epoch_metrics.jsonl"
    expected_epochs = info["config"]["epochs"]
    if metrics.is_file():
        rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row.get("epoch") for row in rows] != list(range(1, expected_epochs + 1)):
            raise ValueError(f"native epoch records incomplete: {metrics}")
        if any(not isinstance(row.get("validation"), dict) for row in rows):
            raise ValueError(f"native validation missing from epoch records: {metrics}")
        pairs = [(row["epoch"], float(row["validation"]["f1"])) for row in rows]
        source = metrics
    else:
        source = Path(info["completion_log"])
        content = source.read_text(encoding="utf-8", errors="replace")
        pattern = re.compile(r"(?m)^Epoch\s+(\d+)\s+finished[^\n]*\nValidation scores:\s*(\{[^}]*\})")
        pairs = [(int(match.group(1)), float(json.loads(match.group(2))["f1"])) for match in pattern.finditer(content)]
        if [epoch for epoch, _ in pairs] != list(range(1, expected_epochs + 1)):
            raise ValueError(f"cannot reconstruct every native dev epoch: {source}")
    epoch, f1 = max(pairs, key=lambda pair: pair[1])
    if abs(f1 - float(info["scores"]["f1"])) > 1e-10:
        raise ValueError(f"best epoch F1 disagrees with saved native validation scores: {source}")
    best_epoch_file = run_dir / "best_epoch.json"
    if best_epoch_file.exists():
        saved = json.loads(best_epoch_file.read_text(encoding="utf-8"))
        if saved.get("epoch") != epoch or abs(float(saved["validation_f1"]) - f1) > 1e-10:
            raise ValueError(f"best_epoch.json conflicts with native dev scores: {run_dir}")
    return {"epoch": epoch, "validation_f1": f1, "evidence": str(source),
            "evidence_sha256": file_sha256(source),
            "epoch_validation_f1": [{"epoch": n, "f1": score} for n, score in pairs]}


def check_final_epoch_only(info, epoch):
    """Verify a transfer checkpoint was evaluated/saved only at final epoch."""
    run_dir = Path(info["path"])
    metrics_file = run_dir / "epoch_metrics.jsonl"
    best_file = run_dir / "best_epoch.json"
    rows = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if [row.get("epoch") for row in rows] != list(range(1, epoch + 1)):
        raise ValueError(f"incomplete transfer epoch records: {metrics_file}")
    if any(row.get("validation") is not None for row in rows[:-1]) or not isinstance(rows[-1].get("validation"), dict):
        raise ValueError(f"transfer did not evaluate exclusively at its final epoch: {metrics_file}")
    saved = json.loads(best_file.read_text(encoding="utf-8"))
    if saved.get("epoch") != epoch or abs(float(saved["validation_f1"]) - info["scores"]["f1"]) > 1e-10:
        raise ValueError(f"saved transfer checkpoint is not the fixed final epoch: {best_file}")
    return {"epoch_metrics.jsonl": file_sha256(metrics_file), "best_epoch.json": file_sha256(best_file)}


def choose(base, alternative):
    """Stable tie rule: retain pre-existing baseline."""
    return "alternate" if alternative["scores"]["f1"] > base["scores"]["f1"] else "base"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List fixed candidates; no writes, GPU, or data access")
    parser.add_argument("--wait-for-lock", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"candidate_specs": SPECS, "domains": DOMAINS,
                          "initial_runs": len(MODELS) * len(DOMAINS),
                          "additional_runs": "up to 18 selected-config seeds plus up to 3 shared3 transfer runs",
                          "selection": "highest seed0 validation entity F1, ties keep base; every epoch checked"}, indent=2))
        return 0
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; refusing CPU fallback")
    QUEUE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (QUEUE / "queue.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | (0 if args.wait_for_lock else fcntl.LOCK_NB))
    except BlockingIOError as error:
        raise RuntimeError("Another B GPU queue holds the lock; retry with --wait-for-lock") from error
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    journal = QUEUE / f"tuning_{run_id}.jsonl"

    def event(kind, **fields):
        record = {"event": kind, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **fields}
        with journal.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def run(model, dataset, seed, source=None, spec=None, eval_every=1, fixed_choice=None):
        spec = spec or SPECS[model]["alternate"]
        if fixed_choice is not None:
            if source not in ("conll2003", "wnut17") or eval_every != spec["epochs"]:
                raise ValueError("fixed-final training requires a single source and final-only validation")
            base = transfer_fixed(source, fixed_choice, spec["epochs"])
        else:
            base = transfer_tuned(source) if source else tuned_path(model, dataset, seed)
        existing = completed(base, spec, seed, dataset=dataset if not source else None, source=source,
                             eval_every=eval_every, final_epoch=spec["epochs"] if fixed_choice is not None else None)
        if existing:
            if fixed_choice is not None:
                check_final_epoch_only(existing, spec["epochs"])
            event("skip_valid", path=existing["path"], f1=existing["scores"]["f1"])
            return existing
        path = next_path(base)
        log = path.with_suffix(".log")
        cmd = command(model, dataset, spec, seed, path, source, eval_every)
        event("task_start", output=str(path), log=str(log), command=cmd)
        env = os.environ.copy()
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                    "TOKENIZERS_PARALLELISM": "false"})
        env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        seen_device = False
        child = None
        try:
            with log.open("x", encoding="utf-8") as file:
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in child.stdout:
                    file.write(line)
                    file.flush()
                    print(line, end="", flush=True)
                    if line.lstrip().startswith("Device:"):
                        seen_device = True
                        device = line.split(":", 1)[1].split(";", 1)[0].strip()
                        if device != "mps":
                            child.terminate()
                            raise RuntimeError(f"Training unexpectedly selected {device!r}")
                code = child.wait()
            if code != 0 or not seen_device:
                raise RuntimeError(f"Training failed with exit={code}, MPS line seen={seen_device}; see {log}")
            info = completed(path, spec, seed, dataset=dataset if not source else None, source=source,
                             eval_every=eval_every, final_epoch=spec["epochs"] if fixed_choice is not None else None)
            if info is None:
                raise RuntimeError(f"Training did not pass validation audit: {path}")
            if fixed_choice is not None:
                check_final_epoch_only(info, spec["epochs"])
            event("task_complete", output=str(path), f1=info["scores"]["f1"])
            return info
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()

    try:
        event("queue_start", candidate_specs=SPECS)
        selected = {}
        for model in MODELS:
            for dataset in DOMAINS:
                baseline = completed(base_path(model, dataset, 0), SPECS[model]["base"], 0, dataset=dataset)
                if not baseline:
                    raise RuntimeError(f"Missing or invalid baseline: {model}/{dataset}/seed0")
                alt = run(model, dataset, 0)
                winner = choose(baseline, alt)
                selected[(model, dataset)] = {"choice": winner, "candidate_spec": SPECS[model][winner],
                    "candidates": {"base": compact(baseline), "alternate": compact(alt)}, "runs": []}
                event("selected", model=model, dataset=dataset, winner=winner,
                      base_f1=baseline["scores"]["f1"], alternate_f1=alt["scores"]["f1"])
        for (model, dataset), result in selected.items():
            for seed in range(3):
                if result["choice"] == "alternate":
                    info = run(model, dataset, seed)
                else:
                    info = completed(base_path(model, dataset, seed), SPECS[model]["base"], seed, dataset=dataset)
                    if not info:
                        raise RuntimeError(f"Missing or invalid original selected seed: {model}/{dataset}/{seed}")
                result["runs"].append({"seed": seed, **compact(info)})

        transfers = {}
        for source in ("conll2003", "wnut17", "merged"):
            if source == "merged":
                baseline = completed(transfer_base(source), SPECS["bert"]["base"], 0, source=source)
                if not baseline:
                    raise RuntimeError(f"Missing or invalid three-type transfer baseline: {source}")
                alt = run("bert", None, 0, source=source)
                winner = choose(baseline, alt)
                candidates = {"base": compact(baseline), "alternate": compact(alt)}
                transfers[source] = {"choice": winner, "candidate_spec": SPECS["bert"][winner],
                                     "candidates": candidates,
                                     "selected_run": compact(alt if winner == "alternate" else baseline),
                                     "selection_rule": "combined-source mapped validation F1"}
            else:
                winner = selected[("bert", source)]["choice"]
                native_info = completed(
                    base_path("bert", source, 0) if winner == "base" else tuned_path("bert", source, 0),
                    SPECS["bert"][winner], 0, dataset=source)
                if not native_info:
                    raise RuntimeError(f"Native selection evidence vanished: {source}")
                epoch_evidence = native_best_epoch(native_info)
                epoch = epoch_evidence["epoch"]
                train_spec = {**SPECS["bert"][winner], "epochs": epoch}
                fixed = run("bert", None, 0, source=source, spec=train_spec,
                            eval_every=epoch, fixed_choice=winner)
                fixed_record = compact(fixed)
                fixed_record["sha256"].update(check_final_epoch_only(fixed, epoch))
                transfers[source] = {"choice": winner, "candidate_spec": SPECS["bert"][winner],
                                     "training_spec": {**train_spec, "eval_every": epoch},
                                     "candidates": {}, "selected_run": fixed_record,
                                     "source_native_best_epoch": epoch_evidence,
                                     "checkpoint_selection": "final_epoch_only",
                                     "selection_rule": "source native-label BERT configuration and best epoch"}
        manifest = {"version": 1, "state": "complete", "policy": {
            "choice": "seed0 best full-validation entity F1; baseline wins ties; best epoch within each run",
            "candidates": "two joint parameter configurations per domain and method, not individual-factor effects",
            "evaluation": "No test inputs or labels used; all selected configurations repeated in seeds 0,1,2",
            "learning_curve_distill": "Preserve independently pre-fixed exploratory learning-curve and matched distillation schedules; they may differ from tuned full-data configurations",
        }, "candidate_specs": SPECS,
            "domain": {f"{model}/{dataset}": result for (model, dataset), result in selected.items()},
            "transfer": transfers}
        output = OUT / "selection.json"
        if output.exists():
            saved = json.loads(output.read_text(encoding="utf-8"))
            if saved != manifest:
                raise FileExistsError(f"Existing selection differs; preserved: {output}")
        else:
            temp = OUT / f"selection_{run_id}.tmp"
            with temp.open("x", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, ensure_ascii=False)
                file.write("\n")
            os.link(temp, output)
            temp.unlink()
        event("queue_complete", manifest=str(output))
        return 0
    except BaseException as error:
        event("queue_error", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
