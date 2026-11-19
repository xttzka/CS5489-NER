"""Classify the 7B zero-shot errors on CoNLL-2003 and WNUT-17 with and without annotation rules."""

import json
import subprocess
import sys

from models.llm.common import DATA, PREDS, PROJECT, RESULTS, SUBSETS, save_json


DATASETS = ("conll2003", "wnut17")
CONDITIONS = {"plain": "qwen7b_k0_plain_test_300", "rules": "qwen7b_k0_guide_test_300"}
OUTPUT = PROJECT / "results" / "error_analysis" / "c"


def relative(path):
    return path.relative_to(PROJECT).as_posix()


def analyze(dataset, method, directory):
    command = [sys.executable, "-B", "-m", "eval.analyze_ner_errors",
               "--gold", relative(DATA / dataset / "test.jsonl"),
               "--pred", relative(PREDS / dataset / method / "0.jsonl"),
               "--subset-ids", relative(SUBSETS / dataset / "test_300.txt"),
               "--output-dir", relative(directory)]
    subprocess.run(command, cwd=PROJECT, check=True, stdout=subprocess.DEVNULL)
    return json.loads((directory / "summary.json").read_text(encoding="utf-8"))


def main():
    report = {}
    for dataset in DATASETS:
        report[dataset] = {}
        for condition, method in CONDITIONS.items():
            summary = analyze(dataset, method, OUTPUT / f"{dataset}_{condition}")
            run = json.loads((RESULTS / f"{method}_{dataset}_test_seed0.json").read_text(encoding="utf-8"))
            report[dataset][condition] = {
                "method": method,
                "f1": summary["exact_match"]["f1"],
                "error_events": summary["error_events"],
                "categories": summary["categories"],
                "absent_from_sentence": run["output"]["absent_from_sentence"],
                "summary": relative(OUTPUT / f"{dataset}_{condition}" / "summary.json"),
            }
    save_json(RESULTS / "error_types.json", report)
    for dataset, conditions in report.items():
        for condition, row in conditions.items():
            print(dataset, condition, f"F1={row['f1']:.2%}", row["categories"], "absent", row["absent_from_sentence"])


if __name__ == "__main__":
    main()
