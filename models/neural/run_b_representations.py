"""Run the four BiLSTM-CRF representation comparisons in a safe sequence.

This queue shares output/b_training_queue/queue.lock with the main B training
queue. It trains on train, selects on validation, and never reads a test split.
Completed jobs are skipped; partial jobs are kept and retried in new folders.
"""

import argparse
import fcntl
import json
import os
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
from models.neural.bilstm_representations import BERT_REVISION, CHAR_DIM, WORD_DIM


QUEUE_DIR = PROJECT / "output" / "b_training_queue"
LOCK_FILE = QUEUE_DIR / "queue.lock"
MODES = ("random", "glove", "glove_char", "bert")
MARKER_NAME = "representation_queue_complete.json"


@dataclass(frozen=True)
class Task:
    name: str
    mode: str
    dataset: str
    seed: int
    train_size: str
    dev_size: int
    epochs: int
    batch_size: int
    bert_batch_size: int
    hidden_size: int
    learning_rate: float
    dropout: float
    device: str
    output_root: Path
    cache_dir: Path

    @property
    def base_output(self):
        return self.output_root / self.name

    def expected_config(self):
        return {
            "mode": self.mode, "dataset": self.dataset, "seed": self.seed,
            "train_size": self.train_size, "dev_size": self.dev_size,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "bert_batch_size": self.bert_batch_size, "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate, "dropout": self.dropout,
            "device": self.device, "lstm_input_dimensions": WORD_DIM + CHAR_DIM,
        }

    def command(self, output_path):
        command = [
            sys.executable, "-u", "-B", "-m", "models.neural.bilstm_representations",
            "--mode", self.mode, "--dataset", self.dataset, "--seed", str(self.seed),
            "--train-size", self.train_size, "--dev-size", str(self.dev_size),
            "--epochs", str(self.epochs), "--batch-size", str(self.batch_size),
            "--bert-batch-size", str(self.bert_batch_size), "--hidden-size", str(self.hidden_size),
            "--learning-rate", str(self.learning_rate), "--dropout", str(self.dropout),
            "--device", self.device, "--output-dir", str(output_path),
        ]
        if self.mode == "bert":
            command.extend(("--bert-cache-dir", str(self.cache_dir)))
        return command


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="conll2003", choices=("conll2003", "wnut17", "mit_restaurant"))
    parser.add_argument("--mode", action="append", choices=MODES, help="Repeat to select modes; default all four")
    parser.add_argument("--seed", action="append", type=int, help="Repeat to select seeds; default 0, 1, 2")
    parser.add_argument("--train-size", default="1.0")
    parser.add_argument("--dev-size", type=int, default=0, help="0 uses the complete validation split")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bert-batch-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", choices=("mps", "cpu"), required=True,
                        help="Explicit device; use cpu only for deliberate small tests")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "output")
    parser.add_argument("--bert-cache-dir", type=Path, default=PROJECT / "output" / "frozen_bert_features")
    parser.add_argument("--limit", type=int, help="At most this many selected tasks")
    parser.add_argument("--max-minutes", type=float,
                        help="Stop starting new jobs after this budget; a running job may finish later")
    parser.add_argument("--wait-for-lock", action="store_true", help="Wait until the shared B queue lock is free")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--list", action="store_true", help="List tasks without running")
    display.add_argument("--dry-run", action="store_true", help="Show commands and completed results without writing")
    args = parser.parse_args(argv)
    if args.dev_size < 0 or min(args.epochs, args.batch_size, args.bert_batch_size, args.hidden_size) < 1:
        parser.error("epochs, batch sizes and hidden-size must be positive; dev-size must be >= 0")
    if args.learning_rate <= 0 or not 0 <= args.dropout < 1:
        parser.error("learning-rate must be positive and dropout must be in [0, 1)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_minutes is not None and args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")
    if args.seed and any(seed < 0 or seed >= 2**32 for seed in args.seed):
        parser.error("seeds must lie in [0, 2**32)")
    return args


def make_tasks(args):
    selected_modes = set(args.mode or MODES)
    seeds = tuple(dict.fromkeys(args.seed if args.seed is not None else (0, 1, 2)))
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT / args.output_root
    cache_dir = args.bert_cache_dir if args.bert_cache_dir.is_absolute() else PROJECT / args.bert_cache_dir
    tasks = []
    for mode in MODES:
        if mode not in selected_modes:
            continue
        for seed in seeds:
            name = f"bilstm_repr_{args.dataset}_{mode}_seed{seed}"
            tasks.append(Task(name, mode, args.dataset, seed, args.train_size, args.dev_size,
                              args.epochs, args.batch_size, args.bert_batch_size, args.hidden_size,
                              args.learning_rate, args.dropout, args.device, output_root, cache_dir))
    return tasks[:args.limit] if args.limit is not None else tasks


def candidate_outputs(task):
    base = task.base_output
    candidates = [base] if base.exists() else []
    if base.parent.exists():
        candidates.extend(sorted(base.parent.glob(base.name + "__attempt_*")))
    return candidates


def next_output(task):
    base = task.base_output
    if not base.exists():
        return base
    number = 2
    while True:
        candidate = base.with_name(base.name + f"__attempt_{number}")
        if not candidate.exists():
            return candidate
        number += 1


def validate_artifacts(task, path, require_marker=True):
    """Check completion evidence, checkpoint, config and exact validation metrics."""
    try:
        if require_marker:
            marker = json.loads((path / MARKER_NAME).read_text(encoding="utf-8"))
            if marker.get("status") != "complete" or marker.get("task") != task.name:
                raise ValueError("completion marker is missing or mismatched")
            if marker.get("expected_config") != task.expected_config():
                raise ValueError("completion marker config differs")
        config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
        for key, expected in task.expected_config().items():
            if config.get(key) != expected:
                raise ValueError(f"configuration mismatch for {key}")
        if task.mode == "bert":
            if config.get("bert_frozen") is not True or config.get("bert_revision") != BERT_REVISION:
                raise ValueError("frozen BERT configuration mismatch")
        elif config.get("static_embeddings_frozen") is not True:
            raise ValueError("static input was not frozen")
        saved = json.loads((path / "validation_scores.json").read_text(encoding="utf-8"))
        predictions = read_tags_by_id(path / "validation_predictions.jsonl")
        rows = load_rows(task.dataset, "validation")
        if task.dev_size:
            rows = rows[:task.dev_size]
        gold = {row["sample_id"]: row["tags"] for row in rows}
        if len(gold) != len(rows):
            raise ValueError("duplicate validation IDs")
        actual = score_ner(gold, predictions)
        for key in ("gold_records", "scored_records", "tp", "fp", "fn"):
            if actual[key] != saved.get(key):
                raise ValueError(f"validation metric mismatch: {key}")
        if abs(actual["f1"] - saved.get("f1", -1)) > 1e-10:
            raise ValueError("validation F1 mismatch")
        best_epoch = saved.get("best_epoch")
        if not isinstance(best_epoch, int) or not 1 <= best_epoch <= task.epochs:
            raise ValueError("invalid best epoch")
        checkpoint = torch.load(path / "best_model.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("best_epoch") != best_epoch:
            raise ValueError("checkpoint best epoch differs")
        if checkpoint.get("config") != config:
            raise ValueError("checkpoint configuration differs")
        weights = checkpoint["model_state_dict"]
        if weights["bilstm.weight_ih_l0"].shape[1] != WORD_DIM + CHAR_DIM:
            raise ValueError("checkpoint BiLSTM input dimension differs")
        if task.mode != "bert" and "word_embedding.weight" not in weights:
            raise ValueError("checkpoint is missing static embeddings")
        return True, {"f1": actual["f1"], "best_epoch": best_epoch}
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def find_completed(task):
    checked = []
    for path in candidate_outputs(task):
        valid, detail = validate_artifacts(task, path)
        if valid:
            return path, detail, checked
        checked.append({"path": str(path), "reason": str(detail)})
    return None, None, checked


def acquire_lock(wait, deadline):
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = LOCK_FILE.open("a+")
    informed = False
    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            if not wait:
                handle.close()
                raise RuntimeError(f"Another B training queue holds {LOCK_FILE}")
            if deadline is not None and time.monotonic() >= deadline:
                handle.close()
                raise TimeoutError("Time budget expired while waiting for the shared training lock")
            if not informed:
                print(f"Waiting for shared B training lock: {LOCK_FILE}", flush=True)
                informed = True
            time.sleep(2)


def main(argv=None):
    args = parse_args(argv)
    tasks = make_tasks(args)
    if args.list:
        for task in tasks:
            print(f"{task.mode:10} {task.name}")
        print(f"Total: {len(tasks)}")
        return 0
    if args.dry_run:
        for task in tasks:
            completed, detail, checked = find_completed(task)
            target = completed or next_output(task)
            status = f'SKIP completed validation F1={detail["f1"]:.4f}' if completed else "RUN"
            print(f"[{status}] {task.name}: {shlex.join(task.command(target))}")
            if checked:
                print(f"  Preserving {len(checked)} prior incomplete/incompatible output(s)")
        print(f"Selected: {len(tasks)}")
        return 0
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("ERROR: MPS is unavailable; refusing to run the MPS queue on CPU.", file=sys.stderr)
        return 2
    if not (PROJECT / "data" / "processed" / args.dataset).is_dir():
        print(f"ERROR: processed dataset missing: {args.dataset}", file=sys.stderr)
        return 2
    started = time.monotonic()
    deadline = started + 60 * args.max_minutes if args.max_minutes else None
    try:
        lock = acquire_lock(args.wait_for_lock, deadline)
    except (RuntimeError, TimeoutError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    event_file = QUEUE_DIR / f"representations_{run_id}.jsonl"
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    stopped = False

    def event(kind, **fields):
        record = {"event": kind, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "elapsed_seconds": round(time.monotonic() - started, 3), **fields}
        with event_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    try:
        event("queue_start", selected=len(tasks), tasks=[task.name for task in tasks], device=args.device)
        for index, task in enumerate(tasks, 1):
            if deadline is not None and time.monotonic() >= deadline:
                event("time_budget_reached", remaining=len(tasks) - index + 1)
                stopped = True
                break
            completed, detail, checked = find_completed(task)
            if completed:
                counts["skipped"] += 1
                event("skipped_complete", task=task.name, output=str(completed), validation=detail)
                continue
            output = next_output(task)
            log_dir = QUEUE_DIR / "representation_logs" / task.name
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{run_id}.log"
            command = task.command(output)
            environment = os.environ.copy()
            environment.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
            environment.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            event("task_start", index=index, task=task.name, output=str(output), log=str(log_path),
                  command=command, preserved_prior_outputs=checked)
            child = None
            try:
                with log_path.open("x", encoding="utf-8") as log:
                    child = subprocess.Popen(command, cwd=PROJECT, env=environment, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                                             start_new_session=True)
                    device_seen = False
                    done_seen = False
                    device_error = None
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                        if line.startswith("Mode:"):
                            device_seen = True
                            if f"Mode: {task.mode}; device: {task.device};" not in line:
                                device_error = f"unexpected child mode/device: {line.strip()}"
                                child.terminate()
                                break
                        if line.strip() == "Done.":
                            done_seen = True
                    return_code = child.wait()
            except KeyboardInterrupt:
                if child is not None and child.poll() is None:
                    child.terminate()
                    child.wait()
                event("task_interrupted", task=task.name, output=str(output), log=str(log_path))
                raise
            except OSError as error:
                counts["failed"] += 1
                event("task_failed", task=task.name, output=str(output), log=str(log_path), error=str(error))
                stopped = True
                break
            if return_code != 0 or not device_seen or not done_seen or device_error:
                counts["failed"] += 1
                event("task_failed", task=task.name, output=str(output), log=str(log_path), return_code=return_code,
                      error=device_error or "child failed, mode/device line missing, or full completion marker absent")
                stopped = True
                break
            valid, detail = validate_artifacts(task, output, require_marker=False)
            if not valid:
                counts["failed"] += 1
                event("task_failed_validation", task=task.name, output=str(output), log=str(log_path), error=detail)
                stopped = True
                break
            marker = {"status": "complete", "task": task.name, "expected_config": task.expected_config(),
                      "validated": detail, "log": str(log_path), "run_id": run_id,
                      "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            with (output / MARKER_NAME).open("x", encoding="utf-8") as file:
                json.dump(marker, file, ensure_ascii=False, indent=2)
            counts["completed"] += 1
            event("task_complete", task=task.name, output=str(output), log=str(log_path), validation=detail)
    except KeyboardInterrupt:
        stopped = True
        event("queue_interrupted")
    finally:
        event("queue_end", **counts, selected=len(tasks), stopped_early=stopped)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    return 1 if stopped or counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
