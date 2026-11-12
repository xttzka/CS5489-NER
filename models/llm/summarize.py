"""Collect member C's result reports into results/C.csv with the team schema."""

import json

from eval.result_tables import RESULTS, label_space, write_csv
from models.llm.common import TYPE_DESCRIPTIONS


SOURCE = RESULTS / "c"
COUNTS = ("tp", "fp", "fn")
RATIOS = ("precision", "recall", "f1")
STANDARD = {"model": "Qwen/Qwen2.5-7B-Instruct", "shots": 10, "selection": "similar", "guideline": False}
SHARED = "LOC+ORG+PER"


def experiment_for(scope):
    if scope == "validation_300":
        return "prompt_selection"
    if scope in {"test_300", "full"}:
        return "baseline"
    if scope.startswith("train_"):
        return "distillation_labels"
    return "robustness"


def describe(report):
    """Return method, condition, training source and configuration of one extraction report."""
    if report["method"] == "GLiNER":
        return "GLiNER", "standard", "", {"model": report["model"], "threshold": report["threshold"]}
    config = {key: report[key] for key in ("model", "shots", "selection", "guideline", "max_new_tokens")}
    size = "7b" if "7B" in report["model"] else "1.5b"
    setting = f"{size},k{report['shots']}" + (f",{report['selection']}" if report["shots"] else "") \
        + (",rules" if report["guideline"] else "")
    if report.get("adapter"):
        return "LoRA-LLM", "standard", "conll2003", {**config, "adapter": report["adapter"]}
    if report.get("wording"):
        setting += f",wording{report['wording']}"
        config["wording"] = report["wording"]
    elif report.get("example_pool_size"):
        setting += f",pool{report['example_pool_size']}"
        config["example_pool_size"] = report["example_pool_size"]
    elif all(report[key] == value for key, value in STANDARD.items()):
        setting = "standard"
    return "LLM", setting, (report["example_source"] or "").split(":")[0], config


def rows_for(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if "scores" not in report:
        return
    method, condition, source, config = describe(report)
    scope = report["scope"]
    schema = report.get("label_scheme", report["dataset"])
    scores = report["scores"]
    shared = scores.get("label_space", "native") != "native"
    base = {
        "member": "C", "method": method, "condition": condition, "train_source": source,
        "dataset": report["dataset"],
        # Derived inputs come from the CoNLL-2003 test split, the labelling sample from MIT training data.
        "split": "train" if scope.startswith("train_") else "test" if report.get("input") else report["split"],
        "scope": scope, "label_space": SHARED if shared else label_space(TYPE_DESCRIPTIONS[schema]),
        "entity_group": "all", "seed": report.get("seed", 0), "fold": "",
        "config_json": json.dumps({**config, "label_scheme": schema}, sort_keys=True, separators=(",", ":")),
    }
    reference = "results/c/" + path.name

    def emit(experiment, kind, metric, value, pointer):
        unit = "ratio" if metric in RATIOS or metric.endswith("_rate") else "count"
        return {**base, "experiment": experiment, "entity_type": kind, "metric": metric, "value": value,
                "unit": unit, "source_ref": f"{reference}#/{pointer}"}

    experiment = experiment_for(scope)
    for metric in (*COUNTS, *RATIOS):
        yield emit(experiment, "all", metric, scores[metric], f"scores/{metric}")
    for kind, values in (report.get("per_type") or scores.get("per_type") or {}).items():
        for metric in (*COUNTS, *RATIOS):
            pointer = ("per_type" if report.get("per_type") else "scores/per_type") + f"/{kind}/{metric}"
            yield emit(experiment, kind, metric, values[metric], pointer)
    for metric, value in report.get("output", {}).items():
        yield emit("llm_output", "all", metric, value, f"output/{metric}")


def main():
    rows = []
    for path in sorted(SOURCE.glob("*.json")):
        if path.name.startswith(("gliner_", "qwen", "lora1.5b")):
            rows.extend(rows_for(path))
    write_csv(RESULTS / "C.csv", rows)


if __name__ == "__main__":
    main()
