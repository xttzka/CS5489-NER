"""Audit and collect completed role-B validation artifacts without using test data.

Each accepted run must have a completion log proving the requested final epoch
and Done, a nonempty checkpoint, full-validation predictions and matching saved
scores. Native, shared-three-type, distillation and representation experiments
are labelled separately. Only validation.jsonl is opened. Exported CSV uses
the team's schema; duplicate semantic result keys are rejected.
"""

import argparse
import json
import re
import statistics
from pathlib import Path
from types import SimpleNamespace

from eval.result_tables import FIELDS, read_csv, validate_rows
from eval.score_ner import read_tags_by_id, score_ner
from eval.score_transfer import MAPPING_PATH, project_tags
from models.neural.score_b import append_csv, output_rows, score_models


PROJECT = Path(__file__).resolve().parents[2]
DATASETS = ("conll2003", "wnut17", "mit_restaurant")
METHOD_NAMES = {"bert": "M4", "bilstm": "M3", "span": "Span"}
REQUIRED_METRICS = ("scored_records", "tp", "fp", "fn", "precision", "recall", "f1")


def queue_logs(root):
    """Map a run directory to queue-start log paths without treating start as success."""
    result = {}
    directory = root / "output" / "b_training_queue"
    journals = list(directory.glob("run_*.jsonl")) + list(directory.glob("representations_*.jsonl"))
    for journal in sorted(journals):
        with journal.open(encoding="utf-8") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    # A concurrently written final line may not yet be complete.
                    continue
                if event.get("event") == "task_start" and event.get("output") and event.get("log"):
                    result.setdefault(Path(event["output"]).resolve(), []).append(Path(event["log"]))
    return result


def completion_evidence(run_dir, config, known_logs):
    expected_epochs = config.get("epochs")
    if type(expected_epochs) is not int or expected_epochs < 1:
        raise ValueError("invalid or absent requested epochs")
    candidates = [run_dir.with_suffix(".log"), *known_logs.get(run_dir.resolve(), [])]
    for path in candidates:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        completed = {int(value) for value in re.findall(
            r"(?m)^Epoch\s+(\d+)(?:\s+finished\s+in|\s+completed\s+in|\s*\|\s*seconds=)", content
        )}
        done = re.search(r"(?m)^Done\.", content) is not None
        if done and set(range(1, expected_epochs + 1)).issubset(completed):
            return str(path)
    raise ValueError("no log proves all requested epochs and final Done")


def describe_run(run_dir, config):
    name = run_dir.name.lower()
    mode = config.get("mode")
    task = config.get("task", "native")
    if mode in {"random", "glove", "glove_char", "bert"}:
        method, experiment = "M3", "representation"
        condition = "bert_frozen" if mode == "bert" else mode
    elif name.startswith("bert_"):
        method = "M4"
        experiment = {"native": "baseline", "transfer": "transfer", "distill": "distill"}.get(task)
        if experiment is None:
            raise ValueError(f"unrecognized BERT task {task!r}")
        condition = config.get("label_source") if task == "distill" else "standard"
    elif name.startswith("bilstm_"):
        method, experiment, condition = "M3", "baseline", "standard"
    elif name.startswith("span_"):
        method, experiment, condition = "Span", "baseline", "standard"
    else:
        raise ValueError("not a recognized B neural run")

    train_size = str(config.get("train_size", ""))
    if not train_size:
        raise ValueError("missing training size")
    dataset = config.get("dataset")
    source = config.get("source") or dataset
    if task == "transfer":
        if source not in {"conll2003", "wnut17", "merged"}:
            raise ValueError("unknown transfer training source")
        dataset = source
    elif task == "distill":
        dataset = source = "mit_restaurant"
        if condition not in {"gold", "pseudo"}:
            raise ValueError("distillation needs gold or pseudo label source")
    elif dataset not in DATASETS:
        raise ValueError("unknown native dataset")
    if experiment == "baseline" and train_size != "1.0":
        experiment = "learning_curve"
        condition = (f"size={train_size};epochs={config.get('epochs')};"
                     f"eval_every={config.get('eval_every', 1)}")
    labels = config.get("labels")
    if not isinstance(labels, list) or "O" not in labels:
        # Span script stores entity types instead of token labels.
        types = config.get("entity_types")
        if not isinstance(types, list) or not types:
            raise ValueError("run has no explicit label schema")
        labels = ["O", *[tag for kind in types for tag in (f"B-{kind}", f"I-{kind}")]]
    types = sorted({label[2:] for label in labels if isinstance(label, str) and label.startswith("B-")})
    if not types or any(f"I-{kind}" not in labels for kind in types):
        raise ValueError("incomplete BIO label schema")
    seed = config.get("seed")
    if type(seed) is not int:
        raise ValueError("missing integer seed")
    return {"method": method, "experiment": experiment, "condition": condition,
            "dataset": dataset, "train_source": source, "task": task,
            "train_size": train_size, "seed": seed, "entity_types": types}


def validation_gold(root, meta):
    if meta["task"] == "transfer":
        datasets = ("conll2003", "wnut17") if meta["train_source"] == "merged" else (meta["dataset"],)
        settings = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["transfer_three_types"]
        result = {}
        for dataset in datasets:
            records = read_tags_by_id(root / "data" / "processed" / dataset / "validation.jsonl")
            if result.keys() & records.keys():
                raise ValueError("duplicate source-validation sample IDs")
            result.update(project_tags(records, settings["maps"][dataset]))
        return result
    return read_tags_by_id(root / "data" / "processed" / meta["dataset"] / "validation.jsonl")


def inspect_run(root, run_dir, known_logs):
    info = {"run": run_dir.name, "path": str(run_dir), "status": "excluded"}
    try:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        meta = describe_run(run_dir, config)
        info.update(meta)
        if config.get("dev_size") != 0:
            raise ValueError("validation subset/smoke run excluded from full-validation results")
        log = completion_evidence(run_dir, config, known_logs)
        checkpoint = run_dir / "best_model.pt"
        if meta["method"] == "M4":
            model_dir = run_dir / "best_model"
            candidates = (model_dir / "model.safetensors", model_dir / "pytorch_model.bin")
            if not (model_dir / "config.json").is_file():
                raise ValueError("missing BERT config")
            checkpoint = next((path for path in candidates if path.is_file()), candidates[0])
        if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000:
            raise ValueError("checkpoint missing or unexpectedly small")
        prediction_path = run_dir / "validation_predictions.jsonl"
        predictions = read_tags_by_id(prediction_path)
        gold = validation_gold(root, meta)
        score = score_ner(gold, predictions)
        saved_path = run_dir / "validation_scores.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        for metric in REQUIRED_METRICS:
            if metric not in saved or abs(float(score[metric]) - float(saved[metric])) > 1e-10:
                raise ValueError(f"saved and recalculated validation {metric} differ")
        info.update(status="complete", completion_log=log, checkpoint=str(checkpoint),
                    prediction=str(prediction_path), scores=score, config=config)
        return info, gold, predictions
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        info["reason"] = str(error)
        return info, None, None


def semantic_key(row):
    return tuple(str(row.get(key, "")) for key in FIELDS
                 if key not in {"value", "unit", "source_ref", "config_json"})


def write_draft(path, accepted):
    lines = [
        "# Role B report draft — validation-stage inventory", "",
        "> This is an automatically generated draft, not a final test report. "
        "No final test-set metric is inferred from validation results.", "",
        "## Methods (supervised models)", "",
        "TODO: Describe BiLSTM-CRF, BERT token classification and Span classification; "
        "specify token/subword alignment, BIO repair, loss, data and model selection.", "",
        "## Main results", "",
        "TODO: Freeze the validation-based model-selection and seed protocol before "
        "evaluating the held-out tests; complete T3 and F2 with compatible scopes.", "",
        "### Current full-validation baselines", "",
        "These are exploratory validation scores, not held-out test scores. "
        "Each entry is the saved best checkpoint of a completed run.", "",
        "| Dataset | Method | Seeds | Validation micro-F1 |", "|---|---|---|---:|",
    ]
    groups = {}
    for info in accepted:
        if info["experiment"] == "baseline" and info["condition"] == "standard":
            groups.setdefault((info["dataset"], info["method"]), []).append(info)
    if not groups:
        lines.append("| TODO | TODO | — | — |")
    for (dataset, method), entries in sorted(groups.items()):
        entries = sorted(entries, key=lambda entry: entry["seed"])
        configs = {json.dumps({key: value for key, value in entry["config"].items()
                               if key not in {"seed", "output_dir"}}, sort_keys=True)
                   for entry in entries}
        if len(configs) != 1:
            for entry in entries:
                lines.append(f"| {dataset} | {method} | {entry['seed']} | "
                             f"{entry['scores']['f1'] * 100:.2f}% (configuration differs; no aggregate) |")
            continue
        values = [entry["scores"]["f1"] * 100 for entry in entries]
        summary = f"{statistics.mean(values):.2f}%"
        if len(values) > 1:
            summary += f" ± {statistics.stdev(values):.2f} pp (sample SD)"
        lines.append(f"| {dataset} | {method} | {', '.join(str(e['seed']) for e in entries)} | {summary} |")
    lines += [
        "", "## Analysis: source of performance", "",
        "TODO: F3(b) requires matched random, GloVe, GloVe+character and frozen-BERT "
        "input runs; F4 requires matched gold entity vectors, t-SNE and clustering summaries.", "",
        "## Analysis: cross-domain performance", "",
        "TODO: Complete the shared PER/LOC/ORG three-source by five-target matrix and F6(a). "
        "Keep source-validation model selection separate from target-test evaluation.", "",
        "## Extensions and handoff", "",
        "TODO: Supply C with same-ID 50/100/1%/10%/100% learning-curve outcomes (F8), "
        "10% BERT and the matched 2,000-sentence pseudo/gold distillation controls.", "",
        "TODO: Produce perturbation/replacement predictions, error breakdowns, "
        "paired bootstrap intervals, runtime measurements, report figures and T3.", "",
        "## Reproducibility and limitations", "",
        "TODO: Document training configurations, checkpoints, vector coverage, seeds, "
        "resources, version differences and any incomplete experiments. "
        "Small-data CoNLL training length was adjusted after observing a seed-0 validation run.", "",
    ]
    with path.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New directory for inventory and per-run validation reports")
    parser.add_argument("--csv", type=Path, help="Optional shared-schema CSV to append to")
    parser.add_argument("--draft", type=Path, help="Optional new English Markdown report draft")
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.resamples < 1:
            raise ValueError("resamples must be positive")
        if args.output_dir.exists() or (args.draft and args.draft.exists()):
            raise FileExistsError("output directory and draft must be new")
        root = args.project_root.resolve()
        logs = queue_logs(root)
        inspected = [inspect_run(root, path, logs) for path in sorted((root / "output").iterdir())
                     if path.is_dir() and (path / "run_config.json").is_file()]
        accepted, rejected, reports, rows, seen = [], [], {}, [], set()
        for info, gold, predictions in inspected:
            if info["status"] != "complete":
                rejected.append(info)
                continue
            identity = (info["method"], info["experiment"], info["condition"], info["train_source"],
                        info["dataset"], info["seed"], "+".join(info["entity_types"]))
            if identity in seen:
                info.update(status="excluded", reason="duplicate experiment identity; do not select retries by score")
                rejected.append(info)
                continue
            seen.add(identity)
            result = score_models(gold, {info["method"]: predictions}, types=info["entity_types"],
                                  resamples=args.resamples, seed=args.bootstrap_seed)
            result["run"] = {key: value for key, value in info.items() if key != "config"}
            result["run_config"] = info["config"]
            report_path = args.output_dir / f"{info['run']}.json"
            context = SimpleNamespace(experiment=info["experiment"], condition=info["condition"],
                                      train_source=info["train_source"], dataset=info["dataset"],
                                      split="validation", scope="full", model_seed=info["seed"],
                                      scoring="transfer_three_types" if info["task"] == "transfer" else "native")
            new_rows = list(output_rows(result, context, report_path))
            config_json = json.dumps({"training": info["config"], "bootstrap": result["bootstrap"],
                                      "scoring": context.scoring}, ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":"))
            for row in new_rows:
                row["config_json"] = config_json
            rows.extend(new_rows)
            reports[info["run"]] = result
            accepted.append(info)
        if args.csv:
            old = list(read_csv(args.csv)) if args.csv.exists() else []
            existing = {semantic_key(row) for row in old}
            collisions = [row for row in rows if semantic_key(row) in existing]
            if collisions:
                raise ValueError(f"CSV already contains {len(collisions)} semantic metric keys; choose a separate file")
            if len({semantic_key(row) for row in rows}) != len(rows):
                raise ValueError("collected runs contain duplicate semantic metric keys")
            list(validate_rows([*old, *rows]))
        args.output_dir.mkdir(parents=True)
        for name, result in reports.items():
            with (args.output_dir / f"{name}.json").open("x", encoding="utf-8") as target:
                json.dump(result, target, ensure_ascii=False, indent=2)
                target.write("\n")
        inventory = {"split": "validation", "accepted": accepted, "excluded": rejected,
                     "resamples": args.resamples, "bootstrap_seed": args.bootstrap_seed,
                     "policy": "Only full source validation. No test set was read or scored."}
        with (args.output_dir / "inventory.json").open("x", encoding="utf-8") as target:
            json.dump(inventory, target, ensure_ascii=False, indent=2)
            target.write("\n")
        if args.csv and rows:
            append_csv(args.csv, rows)
        if args.draft:
            args.draft.parent.mkdir(parents=True, exist_ok=True)
            write_draft(args.draft, accepted)
        print(f"Completed validation runs: {len(accepted)}; excluded/unverified: {len(rejected)}")
        print(f"Inventory and confidence intervals: {args.output_dir}")
        if args.csv:
            print(f"Appended {len(rows)} rows to {args.csv}")
        if args.draft:
            print(f"Report draft: {args.draft}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
