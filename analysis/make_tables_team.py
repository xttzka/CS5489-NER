"""Generate the tables that compare all extractors: results, voting, error overlap and the method comparison."""

import argparse
import csv
import json
from statistics import mean, stdev

from analysis.data_exploration import ROOT
from analysis.make_tables import OUTPUT, shortstack, table


DATASETS = {
    "conll2003": "CoNLL-03", "conllpp": "CoNLL++", "wikigold": "WikiGold",
    "btc": "BTC", "wnut17": "WNUT-17", "mit_restaurant": "MIT Rest.",
}
NATIVE = ("conll2003", "wnut17", "mit_restaurant")
# Name, results CSV, method key, condition and the labelled data the extractor uses.
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
    """Select one row per seed in the label space that the CRF row of the same test set uses."""
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
    """Return one value per seed in percent."""
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
    "M0": "Dictionary", "M1": "Logistic regression", "M2": "CRF", "M3": "BiLSTM-CRF",
    "M4": "BERT", "Span": "Span classifier", "GLiNER": "GLiNER", "LLM": "Qwen2.5-7B, 10 retrieved",
    "LLM-zero": "Qwen2.5-7B, no examples", "LoRA-LLM": "Qwen2.5-1.5B with LoRA",
}
SHORT_NAMES = {"M0": "Dict.", "M1": "LR", "M2": "CRF", "M3": "BiLSTM", "M4": "BERT", "Span": "Span",
               "GLiNER": "GLiNER", "LLM-zero": "Q0", "LLM": "Q10", "LoRA-LLM": "LoRA"}
GROUP_LABELS = {
    "classical": "Classic", "neural": "Neural",
    "cross_crf_bert_llm": "CRF + BERT + LLM", "cross_lstm_bert_llm": "BiLSTM + BERT + LLM",
}


def comparison_rows(rows, team):
    """F1 of every extractor on the comparison columns, None where it was not evaluated."""
    methods = []
    for _, member, method, condition, _ in METHODS:
        name = SHORT_METHOD.get(method, method)
        if method == "LLM":
            name = {"standard": "Qwen2.5-7B, 10 retrieved", "1.5b,k0": "Qwen2.5-1.5B, no examples",
                    "7b,k0": "Qwen2.5-7B, no examples"}[condition]
        cells = []
        for dataset, scope, _ in COMPARISON_COLUMNS:
            # Seed 0 on the full test set, as in the full-test table; the mean of the seeds on the fixed subsets.
            selected = metric_rows(rows, member, method, condition, dataset, scope, "f1", "0" if scope == "full" else None)
            cells.append(mean(float(row["value"]) for row in selected) if selected else None)
        methods.append((name, cells))

    distillation = metric_rows(rows, "B", "M4", "pseudo", "mit_restaurant", "test_300", "f1", "0",
                               experiment="distill")
    if len(distillation) != 1:
        raise ValueError("expected one result of the BERT tagger trained on LLM labels")
    methods.append(("BERT trained on LLM labels", [float(distillation[0]["value"]) if dataset == "mit_restaurant" else None
                                                   for dataset, _, _ in COMPARISON_COLUMNS]))
    for name in ("cross_crf_bert_llm", "cross_lstm_bert_llm"):
        score = team["ensemble"]["groups"][name]["scores"]["f1"]
        methods.append((GROUP_LABELS[name] + " vote", [score if dataset == "conll2003" else None
                                                       for dataset, _, _ in COMPARISON_COLUMNS]))
    return methods


def method_comparison_table(methods):
    rows = [(name, *("--" if value is None else f"{100 * value:.1f}" for value in cells)) for name, cells in methods]
    measured = table("lrrrr", ("Method", *(shortstack(*heading.rsplit(" ", 1))
                                            for _, _, heading in COMPARISON_COLUMNS)), rows,
                     "Entity F1 (%) on the full CoNLL-03 test set and on the fixed 300-sentence subsets of the other "
                     "datasets. BiLSTM-CRF, BERT and the span classifier show seed 0 on CoNLL-03 and the mean of 3 seeds on the subsets. BTC uses the person, location and organization types and "
                     "the extractors trained on CoNLL-03. A dash marks a test set the extractor was not evaluated on.")
    properties = (
        ("Dictionary", "Labelled training split", "Longest exact match against the entity names seen in training"),
        ("Logistic regression", "Labelled training split", "One decision per word from hand-built features"),
        ("CRF", "Labelled training split", "Hand-built features with weights on label transitions"),
        ("BiLSTM-CRF", "Labelled training split", "Sequence decoding over GloVe and character vectors"),
        ("BERT", "Labelled training split", "Word classification over fine-tuned contextual vectors"),
        ("Span classifier", "Labelled training split", "Classifies whole spans over fine-tuned contextual vectors"),
        ("GLiNER", "Type descriptions", "Scores spans against the type descriptions it is given"),
        ("Qwen2.5, no examples", "Type descriptions", "Generates a JSON list of entities for the types it is given"),
        ("Qwen2.5-7B, 10 retrieved", "Labelled sentences to retrieve from", "Adds 10 retrieved examples to the prompt"),
        ("Qwen2.5-1.5B with LoRA", "10\\% of the CoNLL-03 training split", "Generates the entity list after adapter training"),
        ("BERT trained on LLM labels", "2,000 sentences labelled by the LLM", "BERT tagger with the label set of its training data"),
        ("Voting", "The data of its three members", "Keeps an entity that at least two members return"),
    )
    return measured + "\\par\\medskip\n" + table(
        r"p{0.25\linewidth}p{0.30\linewidth}p{0.38\linewidth}",
        ("Method", "Labelled data it needs", "How it decides"), properties,
        "The labelled data each extractor needs and how it decides. Only the extractors that read type "
        "descriptions accept a new entity type without new training.",
    )


def ensemble_table(team):
    report = team["ensemble"]
    rows = [(SHORT_METHOD.get(key, key), "", f"{100 * scores['precision']:.1f}",
             f"{100 * scores['recall']:.1f}", f"{100 * scores['f1']:.1f}", "--")
            for key, scores in report["models"].items()]
    for name, group in report["groups"].items():
        low, high = group["paired_f1_delta_ci95_pp"]
        scores = group["scores"]
        label = GROUP_LABELS.get(name, name)
        rows.append((shortstack(*label.split(" + ")) if " + " in label else label,
                     ", ".join(SHORT_NAMES[member] for member in group["members"]), f"{100 * scores['precision']:.1f}",
                     f"{100 * scores['recall']:.1f}", f"{100 * scores['f1']:.1f}",
                     shortstack(f"{group['f1_delta_vs_best_pp']:+.1f}", f"{{[{low:+.1f}, {high:+.1f}]}}")))
    result = table("llrrrr", ("Method", "Members", "P", "R", "F1", shortstack("Change", "{[95\\% interval]}")),
                   rows, "Entity precision, recall and F1 (%) of each extractor and of four majority votes on the full "
                   "CoNLL-03 test set. Neural methods show seed 0. A vote keeps an entity that at least two of its three "
                   "members return with the same span and type. Change is the F1 difference in points from the best "
                   "member, with a paired sentence-bootstrap 95% interval.")

    methods = list(report["models"])
    overlap = {frozenset(pair["models"]): pair["combined"]["jaccard"]
               for pair in report["pairwise_error_overlap"]}
    matrix = []
    for i, method in enumerate(methods):
        cells = []
        for j, other in enumerate(methods):
            value = overlap.get(frozenset((method, other)))
            cells.append("" if j >= i or value is None else f"{value:.2f}")
        matrix.append((SHORT_NAMES[method], *cells))
    result += "\\par\\medskip\n" + table("l" + "r" * len(methods),
        ("", *(SHORT_NAMES[method] for method in methods)), matrix,
        "Overlap of the errors of two extractors on the full CoNLL-03 test set: the Jaccard index of their sets of "
        "missed and spurious entities. Q0 is Qwen2.5-7B with no examples and Q10 with 10 retrieved examples.")
    return result


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    rows = load()
    team = json.loads((ROOT / "results" / "team_analysis.json").read_text(encoding="utf-8"))
    tables = {"t3_300": subset_table(rows), "t3_full": full_table(rows), "t3_seeds": seed_table(rows),
              "t4_team": ensemble_table(team), "t5": method_comparison_table(comparison_rows(load(baseline_only=False), team))}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for key, content in tables.items():
        (OUTPUT / f"{key}.tex").write_text(content, encoding="utf-8", newline="\n")
        print(OUTPUT / f"{key}.tex")


if __name__ == "__main__":
    main()
