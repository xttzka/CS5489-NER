"""Plan, freeze and execute role-B entity evaluation without test-set tuning.

`plan` and `freeze` read only training/validation artifacts and checkpoints.
`execute` alone opens target test/derived data, and requires an immutable
manifest with checkpoint/config SHA-256 hashes. All predictions use token-only
input; the gold labels are passed solely to the scoring stage. Each complete
task is published as an atomic directory and may be verified/resumed; existing
or incomplete results are never overwritten.
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from eval.score_ner import bio_to_spans, read_subset_ids, read_tags_by_id, score_ner
from eval.score_transfer import MAPPING_PATH, project_tags
from models.neural.collect_b_results import inspect_run, queue_logs, validation_gold
from models.neural.score_b import score_models


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ("conll2003", "wnut17", "mit_restaurant")
ALL_TARGETS = ("conll2003", "conllpp", "wikigold", "btc", "wnut17")
METHODS = ("M3", "M4", "Span")
TRANSFER_TYPES = ("LOC", "ORG", "PER")
SEARCH_SPECS = {
    "bilstm": {"module": "models.neural.bilstm_crf", "base": {"epochs": 5, "batch_size": 16, "learning_rate": .001, "hidden_size": 128, "dropout": .3}, "alternate": {"epochs": 5, "batch_size": 16, "learning_rate": .0005, "hidden_size": 96, "dropout": .5}},
    "bert": {"module": "models.neural.bert_extended", "base": {"epochs": 3, "batch_size": 8, "learning_rate": 3e-5}, "alternate": {"epochs": 3, "batch_size": 16, "learning_rate": 2e-5}},
    "span": {"module": "models.neural.span_bert", "base": {"epochs": 3, "batch_size": 4, "learning_rate": 3e-5}, "alternate": {"epochs": 3, "batch_size": 2, "learning_rate": 2e-5}},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(root, value):
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"path is outside the project: {value}")
    return resolved


def safe_id(text):
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    if not name:
        raise ValueError("empty task identifier")
    return name


def json_hash(data):
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def complete_runs(root):
    logs = queue_logs(root)
    accepted, rejected = [], []
    for directory in sorted((root / "output").iterdir()):
        if directory.is_dir() and (directory / "run_config.json").is_file():
            info, _, _ = inspect_run(root, directory, logs)
            (accepted if info["status"] == "complete" else rejected).append(info)
    return accepted, rejected


def preferred(runs):
    """Choose completed retries by stable path order, never by F1."""
    selected = {}
    for info in runs:
        config = info["config"]
        if info["experiment"] == "baseline":
            key = ("domain", info["method"], info["dataset"], info["seed"])
        elif info["experiment"] == "learning_curve" and info["method"] == "M4":
            size = info["train_size"]
            if size not in ("50", "100", "0.01", "0.1"):
                continue
            # Consistent with the previously documented validation protocol.
            expected = (3, 1) if size == "0.1" else (20, 5)
            if (config.get("epochs"), config.get("eval_every", 1)) != expected:
                continue
            key = ("learning", info["dataset"], size, info["seed"])
        elif info["experiment"] == "transfer":
            key = ("transfer", info["train_source"], info["seed"])
        elif info["experiment"] == "representation":
            key = ("representation", info["dataset"], info["condition"], info["seed"])
        elif info["experiment"] == "distill":
            key = ("distill", info["condition"], info["seed"])
        else:
            continue
        selected.setdefault(key, info)
    return selected


def task_from(info, group, target, scope, experiment, condition="standard", input_path=None,
              target_schema=None, prediction_schema=None):
    source = info["train_source"]
    parts = (group, info["method"], source, target, scope, condition, f"seed{info['seed']}")
    task_id = safe_id("__".join(str(part) for part in parts))
    return {
        "id": task_id, "group": group, "experiment": experiment, "condition": condition,
        "method": info["method"], "run_dir": info["path"], "checkpoint": info["checkpoint"],
        "train_source": source, "target": target, "scope": scope, "seed": info["seed"],
        "input": input_path or f"data/processed/{target}/test.jsonl",
        "subset_ids": f"data/subsets/{target}/test_300.txt" if scope == "test_300" else None,
        "entity_types": info["entity_types"],
        "target_schema": target_schema, "prediction_schema": prediction_schema,
        "mode": info["config"].get("mode"),
    }


def verify_search_evidence(root, evidence_path):
    """Independently re-score every selected and candidate run using dev only."""
    evidence = read_json(evidence_path)
    if evidence.get("version") != 1 or evidence.get("state") != "complete" or evidence.get("candidate_specs") != SEARCH_SPECS:
        raise ValueError("missing or incompatible completed hyperparameter search evidence")
    domains = evidence.get("domain", {})
    expected_keys = {f"{model}/{dataset}" for model in SEARCH_SPECS for dataset in NATIVE}
    if set(domains) != expected_keys or set(evidence.get("transfer", {})) != {"conll2003", "wnut17", "merged"}:
        raise ValueError("search evidence lacks one or more required domain or transfer comparisons")
    logs, cache = queue_logs(root), {}
    methods = {"bilstm": "M3", "bert": "M4", "span": "Span"}

    def check(record, model, dataset, source, candidate, seed, spec_override=None, eval_every=1, extra_files=()):
        if not isinstance(record, dict):
            raise ValueError("malformed search run evidence")
        path = project_path(root, record["run_dir"])
        key = str(path)
        if key not in cache:
            info, _, _ = inspect_run(root, path, logs)
            if info["status"] != "complete":
                raise ValueError(f"search run is incomplete or fails full-validation re-score: {path}")
            cache[key] = info
        info = cache[key]
        config = info["config"]
        spec = spec_override or SEARCH_SPECS[model][candidate]
        if any(config.get(k) != v for k, v in spec.items()) or config.get("seed") != seed or config.get("dev_size") != 0 or str(config.get("train_size")) != "1.0" or config.get("eval_every", 1) != eval_every:
            raise ValueError(f"search run configuration disagrees with prescribed candidate: {path}")
        if config.get("device_used", config.get("device")) != "mps" or info["method"] != methods[model]:
            raise ValueError(f"search run did not use the prescribed method/MPS: {path}")
        if source is None:
            if info["dataset"] != dataset or info["task"] != "native":
                raise ValueError(f"search run is not a native {dataset} model: {path}")
        elif info["task"] != "transfer" or info["train_source"] != source:
            raise ValueError(f"search run is not the expected three-class transfer source: {path}")
        expected_files = {name: sha256(path / name) for name in ("run_config.json", "validation_predictions.jsonl", "validation_scores.json", *extra_files)}
        if record.get("sha256") != expected_files or abs(float(record["validation_f1"]) - info["scores"]["f1"]) > 1e-10:
            raise ValueError(f"search evidence hash or validation F1 disagrees: {path}")
        for field, value in (("validation_predictions", info["prediction"]), ("checkpoint", info["checkpoint"]), ("completion_log", info["completion_log"])):
            if Path(record[field]).resolve() != Path(value).resolve():
                raise ValueError(f"search evidence {field} disagrees with audited run: {path}")
        if record.get("completion_log_sha256") != sha256(info["completion_log"]):
            raise ValueError(f"training completion log changed: {path}")
        return info

    selected, full_baselines = {}, {}
    for model in SEARCH_SPECS:
        for dataset in NATIVE:
            row = domains[f"{model}/{dataset}"]
            if set(row.get("candidates", {})) != {"base", "alternate"}:
                raise ValueError(f"missing comparison for {model}/{dataset}")
            base = check(row["candidates"]["base"], model, dataset, None, "base", 0)
            alternate = check(row["candidates"]["alternate"], model, dataset, None, "alternate", 0)
            winner = "alternate" if alternate["scores"]["f1"] > base["scores"]["f1"] else "base"
            if row.get("choice") != winner or row.get("candidate_spec") != SEARCH_SPECS[model][winner]:
                raise ValueError(f"wrong dev-selected winner for {model}/{dataset}")
            runs = row.get("runs", [])
            if len(runs) != 3 or {r.get("seed") for r in runs} != {0, 1, 2}:
                raise ValueError(f"three selected seeds required for {model}/{dataset}")
            for record in runs:
                seed = record["seed"]
                info = check(record, model, dataset, None, winner, seed)
                if seed == 0 and Path(info["path"]).resolve() != Path(row["candidates"][winner]["run_dir"]).resolve():
                    raise ValueError(f"selected seed0 differs from winner for {model}/{dataset}")
                selected[("domain", methods[model], dataset, seed)] = info
            if model == "bert":
                full_baselines[dataset] = base
    for source, row in evidence["transfer"].items():
        candidate_rows = row.get("candidates", {})
        if source == "merged":
            if set(candidate_rows) != {"base", "alternate"}:
                raise ValueError("merged source requires both mapped-validation candidates")
            base = check(candidate_rows["base"], "bert", None, source, "base", 0)
            alternate = check(candidate_rows["alternate"], "bert", None, source, "alternate", 0)
            winner = "alternate" if alternate["scores"]["f1"] > base["scores"]["f1"] else "base"
            info = check(row["selected_run"], "bert", None, source, winner, 0)
            if Path(info["path"]).resolve() != Path(candidate_rows[winner]["run_dir"]).resolve():
                raise ValueError("selected merged transfer run differs from dev winner")
        else:
            winner = domains[f"bert/{source}"]["choice"]
            if candidate_rows or row.get("checkpoint_selection") != "final_epoch_only":
                raise ValueError(f"single-source transfer must use only a fixed final checkpoint: {source}")
            native = selected[("domain", "M4", source, 0)]
            native_dir = Path(native["path"])
            metrics = native_dir / "epoch_metrics.jsonl"
            if metrics.is_file():
                epoch_rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
                if any(not isinstance(item.get("validation"), dict) for item in epoch_rows):
                    raise ValueError(f"native validation missing from epoch records: {metrics}")
                pairs = [(item["epoch"], float(item["validation"]["f1"])) for item in epoch_rows]
                epoch_evidence_path = metrics
            else:
                epoch_evidence_path = Path(native["completion_log"])
                content = epoch_evidence_path.read_text(encoding="utf-8", errors="replace")
                pattern = re.compile(r"(?m)^Epoch\s+(\d+)\s+finished[^\n]*\nValidation scores:\s*(\{[^}]*\})")
                pairs = [(int(match.group(1)), float(json.loads(match.group(2))["f1"])) for match in pattern.finditer(content)]
            if [number for number, _ in pairs] != list(range(1, native["config"]["epochs"] + 1)):
                raise ValueError(f"native epoch validation history incomplete: {source}")
            best_epoch, best_f1 = max(pairs, key=lambda item: item[1])
            best_file = native_dir / "best_epoch.json"
            if best_file.exists():
                best = read_json(best_file)
                if best.get("epoch") != best_epoch or abs(float(best["validation_f1"]) - best_f1) > 1e-10:
                    raise ValueError(f"native best-epoch artifact disagrees: {source}")
            claimed = {"epoch": best_epoch, "validation_f1": best_f1, "evidence": str(epoch_evidence_path),
                       "evidence_sha256": sha256(epoch_evidence_path),
                       "epoch_validation_f1": [{"epoch": n, "f1": value} for n, value in pairs]}
            if row.get("source_native_best_epoch") != claimed or abs(best_f1 - native["scores"]["f1"]) > 1e-10:
                raise ValueError(f"single-source transfer epoch does not match native dev optimum: {source}")
            train_spec = {**SEARCH_SPECS["bert"][winner], "epochs": best_epoch}
            if row.get("training_spec") != {**train_spec, "eval_every": best_epoch}:
                raise ValueError(f"single-source transfer training schedule differs from native optimum: {source}")
            info = check(row["selected_run"], "bert", None, source, winner, 0,
                         spec_override=train_spec, eval_every=best_epoch,
                         extra_files=("epoch_metrics.jsonl", "best_epoch.json"))
            transfer_dir = Path(info["path"])
            history = [json.loads(line) for line in (transfer_dir/"epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            if ([item.get("epoch") for item in history] != list(range(1, best_epoch + 1))
                    or any(item.get("validation") is not None for item in history[:-1])
                    or not isinstance(history[-1].get("validation"), dict)
                    or abs(float(history[-1]["validation"]["f1"]) - info["scores"]["f1"]) > 1e-10):
                raise ValueError(f"transfer validation must occur only at the fixed final epoch: {source}")
            final = read_json(transfer_dir/"best_epoch.json")
            if final.get("epoch") != best_epoch or abs(float(final["validation_f1"]) - info["scores"]["f1"]) > 1e-10:
                raise ValueError(f"transfer saved checkpoint is not its fixed final epoch: {source}")
        if row.get("choice") != winner or row.get("candidate_spec") != SEARCH_SPECS["bert"][winner]:
            raise ValueError(f"transfer configuration differs from prescribed selection: {source}")
        selected[("transfer", source, 0)] = info
    return selected, full_baselines


def make_plan(root, allow_partial, search_path=None):
    runs, rejected = complete_runs(root)
    chosen = preferred(runs)
    search = None
    learning_baselines = {}
    if search_path:
        search_path = project_path(root, search_path)
        overrides, learning_baselines = verify_search_evidence(root, search_path)
        chosen.update(overrides)
        from models.neural.run_b_representations import find_completed, make_tasks, parse_args
        representation_tasks = make_tasks(parse_args(["--device", "mps"]))
        if len(representation_tasks) != 12:
            raise ValueError("the representation queue must define exactly 12 full configurations")
        logs = queue_logs(root)
        for task in representation_tasks:
            path, detail, _ = find_completed(task)
            mode = "bert_frozen" if task.mode == "bert" else task.mode
            key = ("representation", task.dataset, mode, task.seed)
            if path is None:
                chosen.pop(key, None)
                continue
            info, _, _ = inspect_run(root, path, logs)
            if info["status"] != "complete":
                raise ValueError(f"representation completion marker contradicts audited run: {path}")
            chosen[key] = info
        search = {"path": str(search_path), "sha256": sha256(search_path)}
    elif not allow_partial:
        raise ValueError("a completed, validated hyperparameter search evidence file is required")
    missing = []
    expected = []
    for method in METHODS:
        for dataset in NATIVE:
            for seed in range(3):
                expected.append(("domain", method, dataset, seed))
    for dataset in NATIVE:
        for size, seeds in (("50", range(3)), ("100", range(3)), ("0.01", (0,)), ("0.1", (0,))):
            expected.extend(("learning", dataset, size, seed) for seed in seeds)
    expected.extend(("transfer", source, 0) for source in ("conll2003", "wnut17", "merged"))
    expected.extend(("representation", "conll2003", mode, seed)
                    for mode in ("random", "glove", "glove_char", "bert_frozen")
                    for seed in range(3))
    expected.extend(("distill", mode, 0) for mode in ("gold", "pseudo"))
    missing = [list(key) for key in expected if key not in chosen]
    if missing and not allow_partial:
        raise ValueError(f"{len(missing)} required training configurations are missing; "
                         "complete them before planning, or explicitly select --allow-partial")

    tasks = []
    for key in expected:
        if key not in chosen:
            continue
        info = chosen[key]
        group = key[0]
        if group == "domain":
            _, method, dataset, seed = key
            for scope in ("full", "test_300"):
                tasks.append(task_from(info, "domain", dataset, scope, "baseline"))
            if method == "M4" and seed == 0:
                learning_info = learning_baselines.get(dataset, info)
                tasks.append(task_from(learning_info, "learning", dataset, "test_300", "learning_curve",
                                       condition="size=1.0;epochs=3;eval_every=1"))
            if dataset == "conll2003":
                for target in ("conllpp", "wikigold", "btc"):
                    transfer = {"target_schema": "btc", "prediction_schema": "conll2003"} if target == "btc" else {}
                    tasks.append(task_from(info, "generalization300", target, "test_300", "baseline", **transfer))
                if seed == 0:
                    # Time generalization needs full CoNLL++ with four native types,
                    # not just the fixed-300 or shared-three-type matrix result.
                    tasks.append(task_from(info, "temporal", "conllpp", "full", "temporal"))
                    derived = (
                        ("replacement_original_v2", "data/derived/entity_replacement/conll2003/original_v2.jsonl"),
                        ("replacement_replaced_v2", "data/derived/entity_replacement/conll2003/replaced_v2.jsonl"),
                        ("lowercase_v1", "data/derived/perturbations/conll2003/lowercase_v1.jsonl"),
                        ("spelling_noise_v1", "data/derived/perturbations/conll2003/spelling_noise_v1.jsonl"),
                        ("no_punctuation_v1", "data/derived/perturbations/conll2003/no_punctuation_v1.jsonl"),
                    )
                    for condition, path in derived:
                        tasks.append(task_from(info, "robustness", dataset, "full", "robustness",
                                               condition=condition, input_path=path))
        elif group == "learning":
            _, dataset, size, seed = key
            condition = info["condition"]
            tasks.append(task_from(info, "learning", dataset, "test_300", "learning_curve", condition))
            if dataset == "conll2003" and size == "0.1" and seed == 0:
                for target in ALL_TARGETS:
                    # C reports native four types for these three news/Wiki sets;
                    # BTC and WNUT use the common three-type projection.
                    transfer = ({"target_schema": target, "prediction_schema": "conll2003"}
                                if target in ("btc", "wnut17") else {})
                    tasks.append(task_from(info, "lora_match", target, "test_300", "lora_match",
                                           condition="size_0_1", **transfer))
        elif group == "transfer":
            for target in ALL_TARGETS:
                tasks.append(task_from(info, "transfer", target, "full", "transfer",
                                       target_schema=target, prediction_schema="conll2003"))
        elif group == "representation":
            tasks.append(task_from(info, "representation", "conll2003", "full", "representation",
                                   condition=info["condition"]))
        elif group == "distill":
            for scope in ("full", "test_300"):
                tasks.append(task_from(info, "distill", "mit_restaurant", scope, "distill",
                                       condition=info["condition"]))
    if len({task["id"] for task in tasks}) != len(tasks):
        raise ValueError("task IDs are not unique")
    return {
        "version": 1, "state": "draft", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "search_evidence": search,
        "policy": {
            "main_model": "Not selected by this script; compare all supervised methods on common validation before declaring primary model.",
            "retries": "Tuned domain and transfer winners come from audited dev search; other fixed experiments use first completed configuration in sorted path order. No test-score selection.",
            "learning_curve": "50/100/1% use 20 epochs and evaluation every 5; 10% uses 3 epochs. These are exploratory validation-derived settings.",
            "seeds": "All planned seeds are evaluated; do not select a best seed using test scores.",
            "test": "No test inputs read before an explicit freeze; no parameter changes based on test results.",
        },
        "partial": bool(missing), "missing_requirements": missing,
        "excluded_or_unverified_runs": [{k: v for k, v in row.items() if k != "config"} for row in rejected],
        "tasks": tasks,
    }


def validation_selection(root, path, chosen):
    """Select the highest mean validation F1 across all supervised families.

    M0-M2 require one validated prediction each and M3/M4/Span require all three
    completed domain seeds. This function never opens a test split.
    """
    evidence_file = Path(path)
    evidence = read_json(evidence_file)
    expected = {"dataset": "conll2003", "split": "validation", "scope": "full",
                "label_space": "LOC+MISC+ORG+PER"}
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("selection evidence must declare CoNLL full, native four-type validation")
    if not isinstance(evidence.get("classic_configuration_note"), str) or not evidence["classic_configuration_note"].strip():
        raise ValueError("selection evidence must document the M0–M2 training/feature configuration")
    entries = evidence.get("predictions")
    if not isinstance(entries, list):
        raise ValueError("selection evidence must contain a prediction list")
    by_key = {}
    for row in entries:
        method, seed = row.get("method"), row.get("seed")
        if method not in ("M0", "M1", "M2", "M3", "M4", "Span") or type(seed) is not int:
            raise ValueError("selection prediction has an invalid method or seed")
        key = method, seed
        if key in by_key:
            raise ValueError(f"duplicate selection prediction: {key}")
        by_key[key] = row
    required = {(method, 0) for method in ("M0", "M1", "M2")}
    required |= {(method, seed) for method in METHODS for seed in range(3)}
    if set(by_key) != required:
        raise ValueError(f"selection predictions must contain exactly the full six-family validation set; missing={sorted(required-set(by_key))}")
    gold_path = root / "data" / "processed" / "conll2003" / "validation.jsonl"
    gold = read_tags_by_id(gold_path)
    result = {}
    for (method, seed), row in sorted(by_key.items()):
        path = project_path(root, row["path"])
        digest = sha256(path)
        if row.get("sha256") != digest:
            raise ValueError(f"{method} seed {seed}: selection prediction hash differs from evidence")
        if method in METHODS:
            run = chosen.get(("domain", method, "conll2003", seed))
            if run is None or path != project_path(root, run["prediction"]):
                raise ValueError(f"{method} seed {seed}: selection path is not the completed frozen run's validation predictions")
        predictions = read_tags_by_id(path)
        value = score_ner(gold, predictions)["f1"]
        result.setdefault(method, []).append({"seed": seed, "f1": value, "path": str(path), "sha256": digest})
    order = ("M0", "M1", "M2", "M3", "M4", "Span")
    means = {method: sum(row["f1"] for row in result[method]) / len(result[method]) for method in order}
    # A stable predeclared order breaks exact ties; test scores never participate.
    primary = max(order, key=lambda method: (means[method], -order.index(method)))
    return {"dataset": "conll2003", "split": "validation", "scope": "full",
            "label_space": expected["label_space"], "method_mean_f1": means,
            "predictions": result, "primary_method": primary,
            "primary_seed": 0, "tie_break": list(order),
            "rule": "highest mean native-four-class full-validation F1; classic one seed, neural three; seed 0 representative",
            "classic_configuration_note": evidence["classic_configuration_note"],
            "gold_sha256": sha256(gold_path), "evidence_sha256": sha256(evidence_file)}


def checkpoint_assets(root, info):
    checkpoint = project_path(root, info["checkpoint"])
    paths = {checkpoint}
    if info["method"] == "M4":
        paths.update(path for path in checkpoint.parent.iterdir() if path.is_file())
    elif info["method"] == "Span":
        tokenizer = Path(info["path"]) / "tokenizer"
        if not tokenizer.is_dir():
            raise ValueError(f"missing Span tokenizer: {tokenizer}")
        paths.update(path for path in tokenizer.iterdir() if path.is_file())
    if info.get("experiment") == "representation" and info.get("condition") == "bert_frozen":
        from huggingface_hub import snapshot_download
        from models.neural.bilstm_representations import BERT_NAME, BERT_REVISION
        snapshot = Path(snapshot_download(BERT_NAME, revision=BERT_REVISION, local_files_only=True))
        paths.update(path.resolve() for path in snapshot.rglob("*") if path.is_file())
    return {str(path): sha256(path) for path in sorted(paths)}


def freeze_plan(root, plan, selection_path):
    if plan.get("version") != 1 or plan.get("state") != "draft" or not plan.get("tasks"):
        raise ValueError("expected a nonempty version-1 draft plan")
    if plan.get("partial") or plan.get("missing_requirements"):
        raise ValueError("cannot freeze an incomplete plan for held-out evaluation")
    search = plan.get("search_evidence")
    if (not isinstance(search, dict) or not isinstance(search.get("path"), str) or not search["path"]
            or sha256(project_path(root, search["path"])) != search.get("sha256")):
        raise ValueError("completed hyperparameter search evidence is absent or changed")
    canonical = make_plan(root, allow_partial=False, search_path=search["path"])
    if plan.get("tasks") != canonical["tasks"] or plan.get("policy") != canonical["policy"]:
        raise ValueError("draft plan differs from the current validation-verified complete inventory; regenerate it")
    logs = queue_logs(root)
    runs = {}
    complete, _ = complete_runs(root)
    chosen = preferred(complete)
    overrides, _ = verify_search_evidence(root, search["path"])
    chosen.update(overrides)
    selection = validation_selection(root, selection_path, chosen)
    for task in plan["tasks"]:
        run_dir = project_path(root, task["run_dir"])
        key = str(run_dir)
        if key not in runs:
            info, _, _ = inspect_run(root, run_dir, logs)
            if info["status"] != "complete":
                raise ValueError(f"run is not complete and validation-verified: {run_dir}: {info.get('reason')}")
            checkpoint = project_path(root, info["checkpoint"])
            train_ids = training_ids(root, run_dir)
            validation_ids = set(validation_gold(root, info))
            overlap = train_ids & validation_ids
            if overlap:
                raise ValueError(f"training/validation IDs overlap in {run_dir}: {min(overlap)}")
            runs[key] = {
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "assets": checkpoint_assets(root, info),
                "run_config_sha256": sha256(run_dir / "run_config.json"),
                "validation_predictions_sha256": sha256(run_dir / "validation_predictions.jsonl"),
                "validation_scores_sha256": sha256(run_dir / "validation_scores.json"),
                "training_ids_sha256": hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest(),
                "validation_ids_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest(),
                "validation_f1": info["scores"]["f1"],
            }
        if project_path(root, task["checkpoint"]) != Path(runs[key]["checkpoint"]):
            raise ValueError("plan checkpoint differs from the verified training run")
    code_paths = (
        "models/neural/run_b_evaluation.py", "models/neural/score_b.py", "models/neural/predict_bert.py", "models/neural/token_alignment.py",
        "models/neural/predict_neural.py", "models/neural/bert_baseline.py", "models/neural/bilstm_crf.py",
        "models/neural/span_bert.py", "models/neural/bilstm_representations.py", "models/neural/collect_b_results.py",
        "models/neural/run_b_tuning.py", "models/neural/run_b_representations.py", "models/neural/score_classic_b.py", "models/neural/additional_b_analysis.py",
        "models/neural/handoff_f8_b.py", "models/neural/verify_c_t3_b.py", "models/neural/report_b.py", "models/neural/plot_scores_b.py", "models/neural/plot_overlap_f6b_b.py",
        "models/neural/plot_entity_b.py", "models/neural/export_bert_entities.py", "models/neural/export_glove_entities_b.py",
        "models/neural/vector_manifest_b.py", "models/neural/compare_boundary_b.py", "models/neural/finish_b.py",
        "models/neural/review_misc_clusters_b.py", "analysis/cluster_entities.py",
        "models/neural/amend_alignment_b.py", "models/neural/compare_evaluation_versions_b.py",
        "models/llm/common.py", "eval/score_ner.py",
        "eval/score_transfer.py", "eval/baseline_statistics.py", "data/ner_label_mapping.json",
    )
    frozen = {**plan, "state": "frozen", "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "selection": selection, "code_sha256": {name: sha256(root / name) for name in code_paths},
              "runs": runs, "bootstrap": {"resamples": 1000, "seed": 42}}
    frozen["manifest_sha256"] = json_hash(frozen)
    return frozen


def load_frozen(path):
    frozen = read_json(path)
    digest = frozen.pop("manifest_sha256", None)
    if frozen.get("version") != 1 or frozen.get("state") != "frozen" or digest != json_hash(frozen):
        raise ValueError("invalid or modified frozen manifest")
    frozen["manifest_sha256"] = digest
    return frozen


def read_input(path, ids_path=None, expected_dataset=None):
    """Read gold after freezing; return token-only predictor input and scoring labels."""
    rows = []
    by_id = {}
    with path.open(encoding="utf-8-sig") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id, tokens, tags = row.get("sample_id"), row.get("tokens"), row.get("tags")
            if not isinstance(sample_id, str) or not sample_id or sample_id in by_id:
                raise ValueError(f"{path}:{lineno}: invalid or duplicate sample_id")
            if (not isinstance(tokens, list) or not tokens or
                    any(not isinstance(token, str) or not token.strip() for token in tokens) or
                    not isinstance(tags, list) or len(tokens) != len(tags)):
                raise ValueError(f"{sample_id}: invalid token/gold alignment")
            if expected_dataset and (row.get("dataset") != expected_dataset or row.get("split") != "test"):
                raise ValueError(f"{sample_id}: expected dataset={expected_dataset}, split=test")
            bio_to_spans(tags)
            by_id[sample_id] = (tokens, tags)
            rows.append(sample_id)
    if ids_path:
        rows = read_subset_ids(ids_path)
        if len(rows) != 300:
            raise ValueError(f"fixed evaluation subset must contain exactly 300 IDs: {ids_path}")
        missing = set(rows) - by_id.keys()
        if missing:
            raise ValueError(f"subset ID absent from target data: {min(missing)}")
    if not rows:
        raise ValueError("no evaluation rows")
    clean = [{"sample_id": sample_id, "tokens": by_id[sample_id][0]} for sample_id in rows]
    gold = {sample_id: by_id[sample_id][1] for sample_id in rows}
    return clean, gold


def training_ids(root, run_dir):
    config = read_json(run_dir / "run_config.json")
    ids_file = run_dir / "train_ids.txt"
    if ids_file.exists():
        ids = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate training IDs: {ids_file}")
        return set(ids)
    # Legacy runs predate train_ids.txt. For the implemented native sampler the
    # training split is a safe superset of the selected training IDs.
    source = config.get("dataset")
    if source not in NATIVE:
        raise ValueError(f"missing train_ids.txt for non-native run {run_dir}")
    return set(read_tags_by_id(root / "data" / "processed" / source / "train.jsonl"))


def predictor_command(task, token_path, prediction_path, device):
    method = task["method"]
    model = task["checkpoint"]
    if task["group"] == "representation":
        return [sys.executable, "-u", "-B", "-m", "models.neural.bilstm_representations", "predict",
                "--model", model, "--input", str(token_path), "--output", str(prediction_path),
                "--device", device, "--batch-size", "16"]
    if method == "M4":
        return [sys.executable, "-u", "-B", "-m", "models.neural.predict_neural",
                "--method", "bert", "--model", str(Path(model).parent), "--input", str(token_path),
                "--output", str(prediction_path), "--device", device, "--batch-size", "8",
                "--report", str(prediction_path.with_name("prediction_profile.json"))]
    if method in ("M3", "Span"):
        return [sys.executable, "-u", "-B", "-m", "models.neural.predict_neural",
                "--method", "bilstm" if method == "M3" else "span", "--model", model,
                "--input", str(token_path), "--output", str(prediction_path),
                "--device", device, "--batch-size", "16" if method == "M3" else "4",
                "--report", str(prediction_path.with_name("prediction_profile.json"))]
    raise ValueError(f"unknown model method {method!r}")


def check_device_output(task, output, device):
    if task["group"] == "representation":
        marker = f" on {device}"
    elif task["method"] == "M4":
        marker = f"Device: {device}"
    else:
        marker = f" on {device};"
    if marker not in output:
        raise ValueError(f"predictor did not confirm requested device {device}: {output[-800:]}")


def score_predictions(task, gold, predictions, resamples, seed):
    types = task["entity_types"]
    if task["target_schema"]:
        settings = read_json(MAPPING_PATH)["transfer_three_types"]
        maps = settings["maps"]
        if maps.get(task["target_schema"]) is None or maps.get(task["prediction_schema"]) is None:
            raise ValueError("missing shared-three-type dataset mapping")
        gold = project_tags(gold, maps[task["target_schema"]])
        predictions = project_tags(predictions, maps[task["prediction_schema"]])
        types = list(TRANSFER_TYPES)
    return score_models(gold, {task["method"]: predictions}, types=types,
                        resamples=resamples, seed=seed)


def verify_result(destination, task, frozen_hash, input_hash, subset_hash, checkpoint_hash,
                  resamples, bootstrap_seed):
    try:
        meta = read_json(destination / "metadata.json")
        expected = {"task": task, "frozen_manifest_sha256": frozen_hash,
                    "input_sha256": input_hash, "subset_sha256": subset_hash,
                    "checkpoint_sha256": checkpoint_hash, "resamples": resamples,
                    "bootstrap_seed": bootstrap_seed}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError("metadata provenance differs")
        if sha256(destination / "predictions.jsonl") != meta["prediction_sha256"]:
            raise ValueError("prediction checksum differs")
        if sha256(destination / "scores.json") != meta["scores_sha256"]:
            raise ValueError("score checksum differs")
        if meta.get("profile_sha256") is not None and sha256(destination / "prediction_profile.json") != meta["profile_sha256"]:
            raise ValueError("prediction profile checksum differs")
        read_json(destination / "scores.json")
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"existing result {destination} cannot be safely reused: {error}") from error


def execute_tasks(root, frozen, output_root, groups, max_jobs, max_minutes, device, resamples, bootstrap_seed):
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable: refusing to run the evaluation on CPU")
    if device == "cpu":
        raise ValueError("CPU execution is reserved for synthetic validation tests, not held-out evaluation")
    if frozen.get("bootstrap") != {"resamples": resamples, "seed": bootstrap_seed}:
        raise ValueError("bootstrap parameters must match the frozen evaluation protocol")
    for path, expected in frozen["code_sha256"].items():
        if sha256(project_path(root, path)) != expected:
            raise ValueError(f"frozen inference or scoring code changed: {path}")
    tasks = [task for task in frozen["tasks"] if not groups or task["group"] in groups]
    # Complete full-split jobs first so fixed-300 subsets can reuse identical predictions.
    tasks.sort(key=lambda task: task["scope"] != "full")
    if max_jobs:
        tasks = tasks[:max_jobs]
    if not tasks:
        raise ValueError("no tasks match the requested groups")
    queue = root / "output" / "b_training_queue"
    queue.mkdir(parents=True, exist_ok=True)
    lock = (queue / "queue.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise ValueError("training/evaluation queue already owns the MPS lock") from error
    start_time = time.monotonic()
    verified_runs = {}
    def bind_input(path):
        digest = sha256(path)
        directory = output_root / "input_hashes"
        directory.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(str(path).encode()).hexdigest() + ".json"
        record = directory / name
        value = {"path": str(path), "sha256": digest, "frozen_manifest_sha256": frozen["manifest_sha256"]}
        if record.exists():
            if read_json(record) != value:
                raise ValueError(f"held-out input changed after its first evaluation: {path}")
        else:
            write_new_json(record, value)
        return digest
    try:
        for index, task in enumerate(tasks, start=1):
            if max_minutes and time.monotonic() - start_time >= max_minutes * 60:
                print(f"Time budget reached before task {index}; completed tasks are preserved.", flush=True)
                break
            run_dir = project_path(root, task["run_dir"])
            frozen_run = frozen["runs"].get(str(run_dir))
            if not frozen_run:
                raise ValueError(f"run is absent from frozen checkpoint inventory: {run_dir}")
            checkpoint = project_path(root, task["checkpoint"])
            if str(run_dir) not in verified_runs:
                checks = ((checkpoint, "checkpoint_sha256"),
                          (run_dir / "run_config.json", "run_config_sha256"),
                          (run_dir / "validation_predictions.jsonl", "validation_predictions_sha256"),
                          (run_dir / "validation_scores.json", "validation_scores_sha256"))
                for path, key in checks:
                    if sha256(path) != frozen_run[key]:
                        raise ValueError(f"frozen artifact changed: {path}")
                for path, expected in frozen_run["assets"].items():
                    if sha256(path) != expected:
                        raise ValueError(f"frozen model/tokenizer artifact changed: {path}")
                logs = queue_logs(root)
                info, _, _ = inspect_run(root, run_dir, logs)
                if info["status"] != "complete":
                    raise ValueError(f"frozen training run is no longer valid: {run_dir}")
                if checkpoint_assets(root, info) != frozen_run["assets"]:
                    raise ValueError(f"frozen model or tokenizer asset inventory changed: {run_dir}")
                train_ids = training_ids(root, run_dir)
                validation_ids = set(validation_gold(root, info))
                train_hash = hashlib.sha256("\n".join(sorted(train_ids)).encode()).hexdigest()
                validation_hash = hashlib.sha256("\n".join(sorted(validation_ids)).encode()).hexdigest()
                if train_hash != frozen_run["training_ids_sha256"] or validation_hash != frozen_run["validation_ids_sha256"]:
                    raise ValueError(f"training/validation ID set changed after freezing: {run_dir}")
                verified_runs[str(run_dir)] = (train_ids, validation_ids)
            input_path = project_path(root, task["input"])
            ids_path = project_path(root, task["subset_ids"]) if task["subset_ids"] else None
            input_hash = bind_input(input_path)
            subset_hash = bind_input(ids_path) if ids_path else None
            destination = output_root / "tasks" / task["id"]
            if destination.exists():
                verify_result(destination, task, frozen["manifest_sha256"], input_hash,
                              subset_hash, frozen_run["checkpoint_sha256"], resamples, bootstrap_seed)
                print(f"[{index}/{len(tasks)}] verified existing result: {task['id']}", flush=True)
                continue
            clean, gold = read_input(input_path, ids_path, task["target"])
            if sha256(input_path) != input_hash or (ids_path and sha256(ids_path) != subset_hash):
                raise ValueError("held-out input changed while being read")
            train_ids, validation_ids = verified_runs[str(run_dir)]
            overlap = set(gold) & (train_ids | validation_ids)
            if overlap:
                raise ValueError(f"training or validation/evaluation sample IDs overlap: {min(overlap)}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"[{index}/{len(tasks)}] {task['id']}: {len(clean)} sentences", flush=True)
            with tempfile.TemporaryDirectory(prefix=".staging_", dir=destination.parent) as temp:
                staged = Path(temp) / "result"
                staged.mkdir()
                token_path = staged / "tokens.jsonl"
                with token_path.open("x", encoding="utf-8") as target:
                    for row in clean:
                        target.write(json.dumps(row, ensure_ascii=False) + "\n")
                prediction_path = staged / "predictions.jsonl"
                reused = None
                if task["scope"] == "test_300":
                    candidates = [other for other in frozen["tasks"]
                                  if other["scope"] == "full" and other["run_dir"] == task["run_dir"]
                                  and other["input"] == task["input"] and other["target"] == task["target"]
                                  and other["target_schema"] == task["target_schema"]
                                  and other["prediction_schema"] == task["prediction_schema"]]
                    for other in candidates:
                        other_dir = output_root / "tasks" / other["id"]
                        if other_dir.exists():
                            verify_result(other_dir, other, frozen["manifest_sha256"], input_hash, None,
                                          frozen_run["checkpoint_sha256"], resamples, bootstrap_seed)
                            full_predictions = read_tags_by_id(other_dir / "predictions.jsonl")
                            missing_ids = set(gold) - full_predictions.keys()
                            if missing_ids:
                                raise ValueError(f"full predictions missing subset ID {min(missing_ids)}")
                            with prediction_path.open("x", encoding="utf-8") as target:
                                for row in clean:
                                    sample_id = row["sample_id"]
                                    target.write(json.dumps({"sample_id": sample_id, "tags": full_predictions[sample_id]},
                                                            ensure_ascii=False) + "\n")
                            reused = other["id"]
                            break
                if reused is None:
                    command = predictor_command(task, token_path, prediction_path, device)
                    env = os.environ.copy()
                    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                                "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
                    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
                    lines = []
                    process = subprocess.Popen(command, cwd=root, env=env, text=True, bufsize=1,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    wrong_device = False
                    try:
                        with (staged / "prediction.log").open("x", encoding="utf-8") as log:
                            for line in process.stdout:
                                lines.append(line)
                                log.write(line)
                                log.flush()
                                print(line, end="", flush=True)
                                if task["method"] == "M4" and line.startswith("Device:") and line.strip() != f"Device: {device}":
                                    wrong_device = True
                                    process.terminate()
                                    break
                        returncode = process.wait()
                    except BaseException:
                        process.terminate()
                        process.wait()
                        failure_dir = output_root / "failures"
                        failure_dir.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(prefix=task["id"] + "_", suffix=".log",
                                                         dir=failure_dir, mode="w", encoding="utf-8", delete=False) as failure:
                            failure.writelines(lines)
                            print(f"Interrupted prediction log: {failure.name}", flush=True)
                        raise
                    output = "".join(lines)
                    try:
                        if wrong_device:
                            raise ValueError("predictor attempted a different device; stopped before inference")
                        if returncode:
                            raise ValueError(f"prediction exited {returncode}")
                        check_device_output(task, output, device)
                    except ValueError as error:
                        failure_dir = output_root / "failures"
                        failure_dir.mkdir(parents=True, exist_ok=True)
                        with tempfile.NamedTemporaryFile(prefix=task["id"] + "_", suffix=".log",
                                                         dir=failure_dir, mode="w", encoding="utf-8", delete=False) as failure:
                            failure.write(output)
                            failure_path = failure.name
                        raise ValueError(f"prediction failed for {task['id']}: {error}; log: {failure_path}") from error
                else:
                    (staged / "prediction.log").write_text(f"Predictions filtered from full task {reused}.\n", encoding="utf-8")
                predictions = read_tags_by_id(prediction_path)
                scores = score_predictions(task, gold, predictions, resamples, bootstrap_seed)
                scores["evaluation"] = {"split": "test", "scope": task["scope"],
                                        "target": task["target"], "train_source": task["train_source"],
                                        "experiment": task["experiment"], "condition": task["condition"],
                                        "seed": task["seed"], "target_schema": task["target_schema"],
                                        "prediction_schema": task["prediction_schema"]}
                write_new_json(staged / "scores.json", scores)
                metadata = {"task": task, "frozen_manifest_sha256": frozen["manifest_sha256"],
                            "input_sha256": input_hash, "subset_sha256": subset_hash,
                            "checkpoint_sha256": frozen_run["checkpoint_sha256"],
                            "prediction_sha256": sha256(prediction_path),
                            "scores_sha256": sha256(staged / "scores.json"),
                            "profile_sha256": sha256(staged / "prediction_profile.json") if (staged / "prediction_profile.json").exists() else None,
                            "device": device, "sentences": len(clean),
                            "prediction_reused_from": reused,
                            "resamples": resamples, "bootstrap_seed": bootstrap_seed}
                write_new_json(staged / "metadata.json", metadata)
                token_path.unlink()  # This is a temporary token-only copy, not a deliverable.
                # Atomic publication: a concurrent result is never overwritten.
                if destination.exists():
                    raise FileExistsError(f"result appeared concurrently: {destination}")
                staged.rename(destination)
            print(f"Completed: {destination}", flush=True)
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Draft task inventory from validation-complete runs")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--allow-partial", action="store_true")
    plan.add_argument("--search-evidence", type=Path,
                      help="Completed dev-only candidate comparison and selected seeds")
    freeze = commands.add_parser("freeze", help="Freeze a draft using validation/checkpoint evidence")
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--selection-evidence", type=Path, required=True,
                        help="Native-four-class full-validation predictions for M0-M4 and Span")
    freeze.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("execute", help="Evaluate frozen tasks; --run is required")
    run.add_argument("--frozen", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--group", action="append", choices=("domain", "temporal", "generalization300", "learning", "transfer", "representation", "robustness", "distill", "lora_match"))
    run.add_argument("--max-jobs", type=int)
    run.add_argument("--max-minutes", type=float)
    run.add_argument("--resamples", type=int, default=1000)
    run.add_argument("--bootstrap-seed", type=int, default=42)
    run.add_argument("--device", choices=("mps", "cpu"), default="mps")
    run.add_argument("--run", action="store_true", help="Actually execute; otherwise only display frozen tasks")
    args = parser.parse_args()
    root = ROOT.resolve()
    try:
        if args.command == "plan":
            draft = make_plan(root, args.allow_partial, args.search_evidence)
            write_new_json(args.output, draft)
            print(f"Draft: {args.output}; tasks={len(draft['tasks'])}; missing configurations={len(draft['missing_requirements'])}")
        elif args.command == "freeze":
            frozen = freeze_plan(root, read_json(args.plan), args.selection_evidence)
            write_new_json(args.output, frozen)
            print(f"Frozen: {args.output}; tasks={len(frozen['tasks'])}; digest={frozen['manifest_sha256']}")
        else:
            if (args.max_jobs is not None and args.max_jobs < 1) or (args.max_minutes is not None and args.max_minutes <= 0) or args.resamples < 1:
                raise ValueError("max-jobs, max-minutes and resamples must be positive")
            frozen = load_frozen(args.frozen)
            selected = [task for task in frozen["tasks"] if not args.group or task["group"] in args.group]
            if args.max_jobs:
                selected = selected[:args.max_jobs]
            if not args.run:
                for task in selected:
                    print(f"{task['id']}  {task['run_dir']} -> {task['input']}")
                print(f"Dry run only: {len(selected)} tasks; no test input opened.")
            else:
                execute_tasks(root, frozen, args.output_root, args.group, args.max_jobs, args.max_minutes,
                              args.device, args.resamples, args.bootstrap_seed)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
