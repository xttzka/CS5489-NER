"""Generate T3 for all methods from results/A.csv, B.csv and C.csv."""

import csv
import json
from statistics import mean

from analysis.data_exploration import ROOT
from analysis.make_tables import OUTPUT, table


DATASETS = {
    "conll2003": "CoNLL-03", "conllpp": "CoNLL++", "wikigold": "WikiGold",
    "btc": "BTC", "wnut17": "WNUT-17", "mit_restaurant": "MIT Rest.",
}
NATIVE = ("conll2003", "wnut17", "mit_restaurant")
# Method, result table, method key, condition, label for the training data the extractor uses.
METHODS = (
    ("Dictionary", "A", "M0", "standard", "Full training split"),
    ("Logistic regression", "A", "M1", "standard", "Full training split"),
    ("CRF", "A", "M2", "standard", "Full training split"),
    ("BiLSTM-CRF", "B", "M3", "standard", "Full training split"),
    ("BERT", "B", "M4", "standard", "Full training split"),
    ("Span classifier", "B", "Span", "standard", "Full training split"),
    ("Qwen2.5-1.5B with LoRA", "C", "LoRA-LLM", "standard", "10\\% of CoNLL-03"),
    ("Qwen2.5-7B", "C", "LLM", "standard", "10 retrieved examples"),
    ("GLiNER", "C", "GLiNER", "standard", "None"),
    ("Qwen2.5-1.5B", "C", "LLM", "1.5b,k0", "None"),
    ("Qwen2.5-7B", "C", "LLM", "7b,k0", "None"),
)
CI_NAMES = {"A": ("bootstrap_ci95.f1.0", "bootstrap_ci95.f1.1"), "B": ("f1_ci95_lower", "f1_ci95_upper")}


def load():
    rows = []
    for member in "ABC":
        with (ROOT / "results" / f"{member}.csv").open(encoding="utf-8-sig") as source:
            rows.extend(row for row in csv.DictReader(source)
                        if row["experiment"] == "baseline" and row["split"] == "test"
                        and row["entity_type"] == "all" and row["entity_group"] == "all")
    return rows


def values(rows, member, method, condition, dataset, scope, metric, seed=None):
    """Return one metric per seed of a method on a test set, scored in the labels the other methods use."""
    reference = {row["label_space"] for row in rows if row["member"] == "A" and row["method"] == "M2"
                 and row["dataset"] == dataset and row["scope"] == scope}
    found = [row for row in rows if row["member"] == member and row["method"] == method
             and row["condition"] == condition and row["dataset"] == dataset and row["scope"] == scope
             and row["metric"] == metric and row["label_space"] in reference
             and (seed is None or row["seed"] == seed)]
    by_seed = {}
    for row in found:
        by_seed.setdefault(row["seed"], []).append(row)
    chosen = []
    for group in by_seed.values():
        if len(group) > 1:
            # Of two runs on one test set, keep the one prompted with that dataset's own type names.
            group = [row for row in group if json.loads(row["config_json"]).get("label_scheme") == dataset]
        if len(group) != 1:
            raise ValueError(f"ambiguous rows for {method} {condition} on {dataset}")
        chosen.append(100 * float(group[0]["value"]))
    return chosen


def subset_table(rows):
    lines = []
    for name, member, method, condition, labels in METHODS:
        cells = []
        for dataset in DATASETS:
            scores = values(rows, member, method, condition, dataset, "test_300", "f1")
            cells.append(f"{mean(scores):.1f}" if scores else "--")
        lines.append((name, labels, *cells))
    return table("ll" + "r" * len(DATASETS), ("Method", "Labelled sentences", *DATASETS.values()), lines,
                 "Entity F1 (%) on the fixed 300-sentence test subsets. BiLSTM-CRF, BERT and the "
                 "span classifier show the mean of 3 seeds. Supervised extractors train on CoNLL-03 for the first "
                 "four datasets and on each dataset's own training split for the last two. BTC uses the person, "
                 "location and organization types. A dash marks a test set whose types the extractor was not "
                 "trained to output.")


def full_table(rows):
    lines = []
    for dataset in NATIVE:
        for name, member, method, condition, _ in METHODS[:6]:
            cells = [values(rows, member, method, condition, dataset, "full", metric, "0")
                     for metric in ("precision", "recall", "f1", *CI_NAMES[member])]
            if any(len(cell) != 1 for cell in cells):
                raise ValueError(f"expected one seed-0 row per metric for {name} on {dataset}")
            p, r, f, low, high = (cell[0] for cell in cells)
            lines.append((DATASETS[dataset] if method == "M0" else "", name, f"{p:.1f}", f"{r:.1f}", f"{f:.1f}",
                          f"[{low:.1f}, {high:.1f}]"))
    return table("llrrrr", ("Dataset", "Method", "P", "R", "F1", "95\\% interval"), lines,
                 "Entity precision, recall and F1 (%) on the full test split of each "
                 "dataset, with the sentence-bootstrap 95% interval of F1. Neural methods show seed 0.")


def main():
    rows = load()
    for key, content in {"t3_300": subset_table(rows), "t3_full": full_table(rows)}.items():
        (OUTPUT / f"{key}.tex").write_text(content, encoding="utf-8", newline="\n")
        print(OUTPUT / f"{key}.tex")


if __name__ == "__main__":
    main()
