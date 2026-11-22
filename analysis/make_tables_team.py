"""Generate the team result, ensemble and method-comparison tables."""

import argparse
import csv
from hashlib import sha256
import json
from statistics import mean, stdev

from analysis.data_exploration import ROOT
from analysis.make_tables import OUTPUT, shortstack, table


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


def load(*, baseline_only=True):
    rows = []
    for member in "ABC":
        with (ROOT / "results" / f"{member}.csv").open(encoding="utf-8-sig") as source:
            rows.extend(row for row in csv.DictReader(source)
                        if (not baseline_only or row["experiment"] == "baseline") and row["split"] == "test"
                        and row["entity_type"] == "all" and row["entity_group"] == "all")
    return rows


def metric_rows(rows, member, method, condition, dataset, scope, metric, seed=None, *, experiment="baseline"):
    """Select one source row per seed using the test set's native/shared label space."""
    reference = {row["label_space"] for row in rows if row["member"] == "A" and row["method"] == "M2"
                 and row["dataset"] == dataset and row["scope"] == scope}
    found = [row for row in rows if row["member"] == member and row["method"] == method
             and row["experiment"] == experiment
             and row["condition"] == condition and row["dataset"] == dataset and row["scope"] == scope
             and row["metric"] == metric and row["label_space"] in reference
             and (seed is None or row["seed"] == seed)]
    by_seed = {}
    for row in found:
        by_seed.setdefault(row["seed"], []).append(row)
    chosen = []
    for _, group in sorted(by_seed.items()):
        if len(group) > 1:
            # Of two runs on one test set, keep the one prompted with that dataset's own type names.
            group = [row for row in group if json.loads(row["config_json"]).get("label_scheme") == dataset]
        if len(group) != 1:
            raise ValueError(f"ambiguous rows for {method} {condition} on {dataset}")
        chosen.append(group[0])
    return chosen


def values(rows, member, method, condition, dataset, scope, metric, seed=None):
    """Return one metric per seed in percentage units for the existing T3 tables."""
    return [100 * float(row["value"]) for row in metric_rows(
        rows, member, method, condition, dataset, scope, metric, seed,
    )]


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


def seed_table(rows):
    lines = []
    for dataset in NATIVE:
        for name, member, method, condition, _ in METHODS[3:6]:
            scores = values(rows, member, method, condition, dataset, "full", "f1")
            lines.append((DATASETS[dataset] if method == "M3" else "", name, f"{mean(scores):.1f}", f"{stdev(scores):.1f}"))
    return table("llrr", ("Dataset", "Method", "Mean F1", "SD"), lines,
                 "Entity F1 (%) on the full test split of each dataset, mean and sample SD over 3 training seeds.")


COMPARISON_COLUMNS = (
    ("conll2003", "full", "CoNLL-03 full"),
    ("wnut17", "test_300", "WNUT-17 300"),
    ("mit_restaurant", "test_300", "MIT 300"),
    ("btc", "test_300", "BTC 300"),
)
SHORT_METHOD = {
    "M0": "M0 Dictionary", "M1": "M1 LogReg", "M2": "M2 CRF", "M3": "M3 BiLSTM-CRF",
    "M4": "M4 BERT", "Span": "Span BERT", "GLiNER": "GLiNER", "LLM": "Qwen-7B 10-shot",
    "LLM-zero": "Qwen-7B zero-shot", "LoRA-LLM": "Qwen-1.5B + LoRA",
}
GROUP_LABELS = {
    "classical": "Classical", "neural": "Neural",
    "cross_crf_bert_llm": "CRF + BERT + LLM", "cross_lstm_bert_llm": "BiLSTM + BERT + LLM",
}


def visible_note(text):
    return "\\par\\smallskip{\\footnotesize " + text + "\\par}\n"


def comparison_report(rows, team):
    """Keep the source row beside each measured cell, including its training source."""
    methods = []
    for _, member, method, condition, labels in METHODS:
        name = SHORT_METHOD.get(method, method)
        if method == "LLM":
            name = {"standard": "Qwen-7B 10-shot", "1.5b,k0": "Qwen-1.5B zero-shot",
                    "7b,k0": "Qwen-7B zero-shot"}[condition]
        cells = {}
        for dataset, scope, _ in COMPARISON_COLUMNS:
            selected = metric_rows(rows, member, method, condition, dataset, scope, "f1", "0")
            if len(selected) > 1:
                raise ValueError(f"expected one seed-0 result for {name}/{dataset}/{scope}")
            cells[dataset] = selected[0] if selected else None
        methods.append({"name": name, "member": member, "method": method, "condition": condition,
                        "annotation": labels, "cells": cells})

    distillation = metric_rows(rows, "B", "M4", "pseudo", "mit_restaurant", "test_300", "f1", "0",
                               experiment="distill")
    if len(distillation) != 1:
        raise ValueError("expected the delivered MIT pseudo-label distillation result")
    methods.append({"name": "LLM-distilled BERT", "member": "B", "method": "M4", "condition": "pseudo",
                    "annotation": "2,000 MIT sentences labelled by the zero-shot LLM",
                    "cells": {dataset: distillation[0] if dataset == "mit_restaurant" else None
                              for dataset, _, _ in COMPARISON_COLUMNS}})

    for ensemble_name in ("cross_crf_bert_llm", "cross_lstm_bert_llm"):
        ensemble = team["ensemble"]["groups"][ensemble_name]
        methods.append({"name": GROUP_LABELS[ensemble_name], "member": "team", "method": "Ensemble",
                        "condition": ensemble_name, "annotation": "Constituent training data and labelled retrieval pool",
                        "cells": {dataset: {"value": ensemble["scores"]["f1"], "unit": "ratio",
                                            "train_source": "conll2003", "scope": "full", "seed": "0",
                                            "label_space": "LOC+MISC+ORG+PER",
                                            "source_ref": f"results/team_analysis.json#/ensemble/groups/{ensemble_name}/scores/f1"}
                                  if dataset == "conll2003" else None for dataset, _, _ in COMPARISON_COLUMNS}})
    return {
        "metric": "Exact entity-level F1",
        "seed": 0,
        "source_tables": {f"results/{member}.csv": sha256((ROOT / "results" / f"{member}.csv").read_bytes()).hexdigest()
                          for member in "ABC"},
        "columns": [{"dataset": dataset, "scope": scope, "heading": heading}
                    for dataset, scope, heading in COMPARISON_COLUMNS],
        "comparison_scope": "Native target types on CoNLL-2003, WNUT-17 and MIT. BTC has shared PER/LOC/ORG types.",
        "training_policy": "Supervised native results train on the corpus's own labelled split. BTC uses CoNLL-2003 "
                           "supervision; zero-shot methods use type descriptions; few-shot methods retrieve 10 "
                           "examples per input from the source labelled pool. The pseudo-label model trains on MIT only.",
        "unmeasured": "A null cell is an unavailable evaluation for that trained model/configuration.",
        "methods": methods,
    }


def method_comparison_table(report):
    rows = []
    for model in report["methods"]:
        cells = ["--" if model["cells"][dataset] is None else f"{100 * float(model['cells'][dataset]['value']):.1f}"
                 for dataset, _, _ in COMPARISON_COLUMNS]
        rows.append((model["name"], *cells))
    measured = table("lrrrr", ("Method", *(shortstack(*heading.rsplit(" ", 1))
                                            for _, _, heading in COMPARISON_COLUMNS)), rows,
                     "T5(a). Empirical accuracy on the stated full or fixed-300 test scope; seed 0 for neural methods.")
    measured += visible_note(
        "(a) Entity F1 (\\%). The columns use their stated sentence sets and native types. BTC uses "
        "PER/LOC/ORG. Supervised WNUT and MIT rows train on each corpus's own labelled training split; "
        "the BTC rows transfer the CoNLL-trained model. LoRA is trained on 10\\% of CoNLL only. "
        "Few-shot prompts retrieve 10 labelled examples per input from the corresponding training pool "
        "(CoNLL for BTC). Neural rows use seed 0. A dash indicates an unavailable evaluation. "
        r"Each measured cell is traceable in \texttt{\detokenize{results/team_method_comparison.json}}."
    )

    properties = (
        ("M0 Dictionary", "Labelled names from training", "Exact longest name match, with dictionary entries for each type"),
        ("M1 LogReg", "Labelled training split", "Per-token feature weights with types fixed by training labels"),
        ("M2 CRF", "Labelled training split", "Token and label-transition weights with trained output types"),
        ("M3 BiLSTM-CRF", "Labelled training split", "Contextual sequence decoding with trained output types"),
        ("M4 BERT", "Labelled training split", "Contextual token classification with trained output types"),
        ("Span BERT", "Labelled training split", "Classifies complete spans with trained output types"),
        ("GLiNER", "Type descriptions at inference", "Scores spans against supplied type names"),
        ("Qwen zero-shot", "Type descriptions at inference", "Generates entities in JSON for supplied types"),
        ("Qwen-7B 10-shot", "Labelled retrieval pool", "Uses 10 retrieved examples per input and supplied type names"),
        ("Qwen-1.5B + LoRA", "10\\% labelled CoNLL split", "Generative extraction after news-data adaptation"),
        ("LLM-distilled BERT", "2,000 LLM-labelled MIT sentences", "Fixed MIT classifier after pseudo-label training"),
        ("Cross-paradigm voting", "All constituent training inputs", "Strict majority over identical typed spans in the common output types"),
    )
    measured += "\\par\\medskip\n" + table(
        r"p{0.21\linewidth}p{0.28\linewidth}p{0.43\linewidth}",
        ("Method", "Annotation supplied", "Decision rule and type adaptation"), properties,
        "T5(b). Structural properties of the implemented extractors. Upstream pretrained models already use training data.",
    )
    return measured + visible_note(
        "(b) Annotation refers to the project data supplied for fitting or prompting. These are properties "
        "of the implemented interfaces. Passing new type descriptions to GLiNER or Qwen makes those "
        "types available at inference. The measured MIT accuracy is in (a). Decoding and type "
        "interfaces are documented by the delivered implementations."
    )


def ensemble_table(team):
    report = team["ensemble"]
    rows = [(SHORT_METHOD.get(key, key), "Single", f"{100 * scores['precision']:.2f}",
             f"{100 * scores['recall']:.2f}", f"{100 * scores['f1']:.2f}", "--")
            for key, scores in report["models"].items()]
    for name, group in report["groups"].items():
        low, high = group["paired_f1_delta_ci95_pp"]
        scores = group["scores"]
        label = GROUP_LABELS.get(name, name)
        rows.append((shortstack(*label.split(" + ")) if " + " in label else label,
                     ", ".join(group["members"]), f"{100 * scores['precision']:.2f}",
                     f"{100 * scores['recall']:.2f}", f"{100 * scores['f1']:.2f}",
                     shortstack(f"{group['f1_delta_vs_best_pp']:+.2f}", f"{{[{low:+.2f}, {high:+.2f}]}}")))
    result = table("llrrrr", ("Method / group", "Members", "P", "R", "F1", shortstack("Change (pp)", "{[95\\% interval]}")),
                   rows, "T4 team ensemble results. CoNLL-2003 complete test; exact boundaries and four native types.")
    result += visible_note(
        "Entity P/R/F1 (\\%) on all 3,453 CoNLL-2003 test sentences. Neural models use seed 0. "
        "The LLM uses the validation-selected 7B, 10-retrieved-example configuration. Groups are fixed "
        "before scoring; strict span-and-type majority is required. Changes compare each ensemble "
        "with its highest-scoring constituent on this test set. Intervals use 1,000 paired "
        "sentence-bootstrap draws. The observed best constituent is held fixed during resampling."
    )

    codes = {"LLM-zero": "Q0", "LLM": "Q10", "LoRA-LLM": "LoRA"}
    methods = list(report["models"])
    overlap = {frozenset(pair["models"]): pair["combined"]["jaccard"]
               for pair in report["pairwise_error_overlap"]}
    matrix = []
    for i, method in enumerate(methods):
        cells = []
        for j, other in enumerate(methods):
            value = overlap.get(frozenset((method, other)))
            cells.append("--" if j >= i or value is None else f"{value:.3f}")
        matrix.append((codes.get(method, method), *cells))
    result += "\\par\\medskip\n" + table("l" + "r" * len(methods),
        ("Error Jaccard", *(codes.get(method, method) for method in methods)), matrix,
        "T4 team lower-triangle error-overlap matrix. Exact typed-span missed and spurious errors are pooled.")
    return result + visible_note(
        "Jaccard is the intersection divided by the union of two methods' exact typed-span error sets, "
        "including sample IDs. Lower values indicate less overlap. Q0 is zero-shot Qwen-7B. "
        "Q10 is the selected few-shot Qwen-7B. LoRA is adapted Qwen-1.5B. The JSON separately records "
        "missed-entity and spurious-entity overlap."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-analysis", action="store_true", help="Also build T4 and T5 from the completed team analysis")
    args = parser.parse_args()
    rows = load()
    tables = {"t3_300": subset_table(rows), "t3_full": full_table(rows), "t3_seeds": seed_table(rows)}
    if args.include_analysis:
        team = json.loads((ROOT / "results" / "team_analysis.json").read_text(encoding="utf-8"))
        comparison = comparison_report(load(baseline_only=False), team)
        (ROOT / "results" / "team_method_comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n",
        )
        tables.update(t5=method_comparison_table(comparison), t4_team=ensemble_table(team))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for key, content in tables.items():
        (OUTPUT / f"{key}.tex").write_text(content, encoding="utf-8", newline="\n")
        print(OUTPUT / f"{key}.tex")


if __name__ == "__main__":
    main()
