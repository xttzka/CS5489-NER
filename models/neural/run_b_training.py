"""Sequential, MPS-only queue for the missing role-B model training runs.

This queue never reads a test split or deletes/replaces existing training
artifacts. Use --list or --dry-run to inspect the exact work first.
"""

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

from eval.score_ner import read_tags_by_id, score_ner
from models.llm.common import PROJECT, load_rows
from models.neural.bert_extended import transfer_rows


@dataclass(frozen=True)
class Task:
    name: str
    group: str
    model: str
    module: str
    output: str
    arguments: tuple
    seed: int
    epochs: int
    batch_size: int
    dataset: str = ""
    source: str = ""
    train_size: str = "1.0"
    eval_every: int = 1
    label_source: str = ""

    def command(self, output_path):
        return [
            sys.executable, "-u", "-B", "-m", self.module,
            *self.arguments, "--seed", str(self.seed), "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size), "--dev-size", "0",
            "--output-dir", str(output_path),
        ]


def make_tasks():
    tasks = []
    for model in ("bert", "bilstm"):
        for dataset in ("wnut17", "mit_restaurant"):
            for seed in (1, 2):
                name = f"{model}_{dataset}_full_seed{seed}"
                if model == "bert":
                    module = "models.neural.bert_extended"
                    args = ("--task", "native", "--dataset", dataset, "--train-size", "1.0",
                            "--device", "mps", "--local-files-only")
                    epochs, batch_size = 3, 8
                else:
                    module = "models.neural.bilstm_crf"
                    args = ("--dataset", dataset, "--train-size", "1.0")
                    epochs, batch_size = 5, 16
                tasks.append(Task(name, "domain", model, module, f"output/{name}", args,
                                  seed, epochs, batch_size, dataset=dataset))

    for dataset, seeds in (("conll2003", (1, 2)), ("wnut17", (0, 1, 2)),
                           ("mit_restaurant", (0, 1, 2))):
        for seed in seeds:
            name = f"span_{dataset}_full_seed{seed}"
            args = ("--dataset", dataset, "--train-size", "1.0", "--max-span-length", "14",
                    "--negative-ratio", "10")
            tasks.append(Task(name, "domain", "span", "models.neural.span_bert", f"output/{name}",
                              args, seed, 3, 4, dataset=dataset))

    for dataset in ("wnut17", "mit_restaurant"):
        for size, size_name, seeds, epochs, evaluate_every in (
            ("50", "50sent_long", (0, 1, 2), 20, 5),
            ("100", "100sent_long", (0, 1, 2), 20, 5),
            ("0.01", "1pct_long", (0,), 20, 5),
            ("0.1", "10pct", (0,), 3, 1),
        ):
            for seed in seeds:
                name = f"bert_{dataset}_{size_name}_seed{seed}"
                args = ("--task", "native", "--dataset", dataset, "--train-size", size,
                        "--eval-every", str(evaluate_every), "--device", "mps", "--local-files-only")
                tasks.append(Task(name, "learning", "bert", "models.neural.bert_extended",
                                  f"output/{name}", args, seed, epochs, 8, dataset=dataset,
                                  train_size=size, eval_every=evaluate_every))

    for source in ("conll2003", "wnut17", "merged"):
        name = f"bert_transfer_{source}_seed0"
        args = ("--task", "transfer", "--source", source, "--train-size", "1.0",
                "--device", "mps", "--local-files-only")
        tasks.append(Task(name, "transfer", "bert", "models.neural.bert_extended",
                          f"output/{name}", args, 0, 3, 8, source=source))

    prediction_file = "preds/mit_restaurant/qwen7b_k0_plain_train_2000_v1/0.jsonl"
    for label_source in ("pseudo", "gold"):
        name = f"bert_distill_mit_{label_source}_seed0"
        args = ("--task", "distill", "--pseudo-predictions", prediction_file,
                "--label-source", label_source, "--device", "mps", "--local-files-only")
        tasks.append(Task(name, "distill", "bert", "models.neural.bert_extended",
                          f"output/{name}", args, 0, 3, 8, dataset="mit_restaurant",
                          label_source=label_source))
    return tasks


def candidate_outputs(task):
    base = PROJECT / task.output
    paths = [base] if base.exists() else []
    paths.extend(sorted(base.parent.glob(base.name + "__attempt_*")))
    return paths


def completed_training_log(task, output_path):
    """Find a log bound to this output with every epoch and the final Done.

    A saved best checkpoint can be produced after epoch 1, so its existence is
    not evidence that the configured training run actually finished. Queue
    journals associate each retry's exact output directory with its own log.
    Previously run manual experiments may instead have output/<name>.log.
    """
    target = output_path.resolve()
    logs = [Path(str(output_path) + ".log")]
    queue_dir = PROJECT / "output" / "b_training_queue"
    for journal in sorted(queue_dir.glob("run_*.jsonl")):
        try:
            with journal.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        record = json.loads(line)
                        if (record.get("event") == "task_start" and record.get("task") == task.name
                                and Path(record["output"]).resolve() == target):
                            logs.append(Path(record["log"]))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        # A live journal can end in a partially written line.
                        continue
        except OSError:
            continue
    epoch_pattern = re.compile(r"^Epoch\s+(\d+)\s+(?:finished|completed)\b", re.MULTILINE)
    done_pattern = re.compile(r"^Done\.(?:\s|$)", re.MULTILINE)
    for log in logs:
        try:
            text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        epochs = list(epoch_pattern.finditer(text))
        done = list(done_pattern.finditer(text))
        seen = {int(match.group(1)) for match in epochs}
        final = [match for match in epochs if int(match.group(1)) == task.epochs]
        if set(range(1, task.epochs + 1)).issubset(seen) and final and done and done[-1].start() > final[-1].start():
            return log
    return None


def validate_completed(task, path):
    """Require configuration, complete dev predictions, and a saved checkpoint."""
    try:
        config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
        scores = json.loads((path / "validation_scores.json").read_text(encoding="utf-8"))
        predicted = read_tags_by_id(path / "validation_predictions.jsonl")
        if task.model == "bert":
            model_dir = path / "best_model"
            weights = [model_dir / "model.safetensors", model_dir / "pytorch_model.bin"]
            if not (model_dir / "config.json").is_file() or not any(
                weight.is_file() and weight.stat().st_size > 1_000_000 for weight in weights
            ):
                raise ValueError("missing BERT checkpoint")
        elif not (path / "best_model.pt").is_file() or (path / "best_model.pt").stat().st_size < 1_000_000:
            raise ValueError("missing model checkpoint")
        required = {"seed": task.seed, "epochs": task.epochs, "batch_size": task.batch_size, "dev_size": 0}
        for key, value in required.items():
            if config.get(key) != value:
                raise ValueError(f"configuration mismatch: {key}")
        if config.get("device_used", config.get("device")) != "mps":
            raise ValueError("training configuration does not identify MPS")
        saved_output = Path(config.get("output_dir", ""))
        if not saved_output.is_absolute():
            saved_output = PROJECT / saved_output
        if saved_output.resolve() != path.resolve():
            raise ValueError("training output directory in configuration does not match")
        if task.group in ("domain", "learning"):
            if config.get("dataset") != task.dataset or config.get("train_size") != task.train_size:
                raise ValueError("dataset or train-size mismatch")
            if task.model == "bert" and config.get("task", "native") != "native":
                raise ValueError("not a native-label model")
            if task.group == "learning" and config.get("eval_every") != task.eval_every:
                raise ValueError("evaluation interval mismatch")
        elif task.group == "transfer":
            if config.get("task") != "transfer" or config.get("source") != task.source:
                raise ValueError("transfer task/source mismatch")
        else:
            if config.get("task") != "distill" or config.get("label_source") != task.label_source:
                raise ValueError("distillation label source mismatch")
        if task.group == "transfer":
            rows = transfer_rows(task.source, "validation", "1.0", task.seed)
        else:
            rows = load_rows(task.dataset, "validation")
        gold = {row["sample_id"]: row["tags"] for row in rows}
        if len(gold) != len(rows):
            raise ValueError("duplicate validation IDs")
        actual = score_ner(gold, predicted)
        for key in ("scored_records", "tp", "fp", "fn"):
            if actual[key] != scores.get(key):
                raise ValueError(f"validation metric mismatch: {key}")
        if abs(actual["f1"] - scores.get("f1", -1)) > 1e-10:
            raise ValueError("validation F1 mismatch")
        if completed_training_log(task, path) is None:
            raise ValueError("no matching training log confirms all epochs and final Done")
        return True, actual["f1"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        return False, str(error)


def next_output(task):
    base = PROJECT / task.output
    if not base.exists():
        return base
    number = 2
    while True:
        candidate = base.with_name(base.name + f"__attempt_{number}")
        if not candidate.exists():
            return candidate
        number += 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", choices=("all", "domain", "learning", "transfer", "distill"))
    parser.add_argument("--task", action="append", help="Exact task name; may be repeated")
    parser.add_argument("--limit", type=int, help="At most this many selected tasks")
    parser.add_argument("--max-minutes", type=float, help="Do not start a new task once this time budget is reached")
    parser.add_argument("--list", action="store_true", help="List selected tasks without executing")
    parser.add_argument("--dry-run", action="store_true", help="Print exact commands without writing files or using MPS")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")
    if args.list and args.dry_run:
        parser.error("choose either --list or --dry-run")
    return parser, args


def select_tasks(parser, args):
    all_tasks = make_tasks()
    by_name = {task.name: task for task in all_tasks}
    if args.task:
        unknown = set(args.task) - by_name.keys()
        if unknown:
            parser.error(f"unknown task names: {', '.join(sorted(unknown))}")
    groups = set(args.group or ["all"])
    selected = [task for task in all_tasks if ("all" in groups or task.group in groups)
                and (not args.task or task.name in args.task)]
    if not selected:
        parser.error("the selected groups and task names have no tasks in common")
    return selected[:args.limit] if args.limit else selected


def find_completed(task):
    checked = []
    for path in candidate_outputs(task):
        valid, detail = validate_completed(task, path)
        if valid:
            return path, detail, checked
        checked.append({"path": str(path), "reason": str(detail)})
    return None, None, checked


def main():
    parser, args = parse_args()
    tasks = select_tasks(parser, args)
    if args.list:
        for task in tasks:
            print(f"{task.group:8} {task.name}")
        print(f"Total: {len(tasks)}")
        return 0
    if args.dry_run:
        for task in tasks:
            completed, score, checked = find_completed(task)
            out = completed or next_output(task)
            status = f"SKIP valid F1={score:.4f}" if completed else "RUN"
            print(f"[{status}] {task.name}: {shlex.join(task.command(out))}")
            if checked:
                print(f"  Existing incomplete or incompatible outputs preserved: {len(checked)}")
        print(f"Selected: {len(tasks)}")
        return 0

    # This must be checked before creating queue files or launching a child.
    if not torch.backends.mps.is_available():
        print("ERROR: Apple MPS is unavailable; refusing to run the training queue on CPU.", file=sys.stderr)
        return 2
    if not (PROJECT / "data" / "processed").is_dir():
        print("ERROR: processed data directory is missing.", file=sys.stderr)
        return 2

    queue_dir = PROJECT / "output" / "b_training_queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (queue_dir / "queue.lock").open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: another B training queue is already running.", file=sys.stderr)
        lock_file.close()
        return 2

    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    event_file = queue_dir / f"run_{run_id}.jsonl"
    started = time.monotonic()
    counts = {"success": 0, "skipped": 0, "failed": 0}

    def event(kind, **fields):
        record = {"event": kind, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "elapsed_seconds": round(time.monotonic() - started, 3), **fields}
        with event_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    event("queue_start", selected=len(tasks), groups=args.group, tasks=[task.name for task in tasks])
    stopped_early = False
    try:
        for index, task in enumerate(tasks, start=1):
            if args.max_minutes is not None and time.monotonic() - started >= args.max_minutes * 60:
                event("time_budget_reached", remaining=len(tasks) - index + 1)
                stopped_early = True
                break
            completed, score, checked = find_completed(task)
            if completed:
                counts["skipped"] += 1
                event("skipped_complete", task=task.name, group=task.group, output=str(completed), validation_f1=score)
                continue
            output_path = next_output(task)
            log_dir = queue_dir / "logs" / task.name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{run_id}.log"
            command = task.command(output_path)
            environment = os.environ.copy()
            environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
            environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            event("task_start", index=index, task=task.name, group=task.group, output=str(output_path),
                  log=str(log_path), command=command, preserved_prior_outputs=checked)
            device_seen = False
            device_error = None
            child = None
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                        if line.lstrip().startswith("Device:"):
                            device_seen = True
                            reported_device = line.split(":", 1)[1].split(";", 1)[0].strip()
                            if reported_device != "mps":
                                device_error = f"child selected non-MPS device: {line.strip()}"
                                child.terminate()
                                break
                    return_code = child.wait()
            except KeyboardInterrupt:
                if child is not None and child.poll() is None:
                    child.terminate()
                    child.wait()
                event("task_interrupted", task=task.name, output=str(output_path), log=str(log_path))
                raise
            except OSError as error:
                counts["failed"] += 1
                event("task_failed", task=task.name, output=str(output_path), log=str(log_path), error=str(error))
                stopped_early = True
                break
            if return_code != 0 or not device_seen or device_error:
                counts["failed"] += 1
                event("task_failed", task=task.name, output=str(output_path), log=str(log_path),
                      return_code=return_code, error=device_error or ("MPS device line missing" if not device_seen else "child process failed"))
                stopped_early = True
                break
            valid, detail = validate_completed(task, output_path)
            if not valid:
                counts["failed"] += 1
                event("task_failed_validation", task=task.name, output=str(output_path), log=str(log_path), error=str(detail))
                stopped_early = True
                break
            counts["success"] += 1
            event("task_complete", task=task.name, group=task.group, output=str(output_path),
                  log=str(log_path), validation_f1=detail)
    except KeyboardInterrupt:
        stopped_early = True
        event("queue_interrupted")
    finally:
        event("queue_end", **counts, selected=len(tasks), stopped_early=stopped_early)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
    return 1 if counts["failed"] or stopped_early else 0


if __name__ == "__main__":
    raise SystemExit(main())
