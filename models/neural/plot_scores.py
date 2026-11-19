"""Draw report figures from the results CSVs: f2 main results, f3b input representations, f6a transfer matrix.

A required value that is missing or duplicated in the CSVs is an error.
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import style
from eval.result_tables import read_csv


NAMES = ("M0", "M1", "M2", "M3", "M4", "Span")
METHOD_ALIASES = {
    "m0": "M0", "dictionary": "M0", "m0 dictionary": "M0",
    "m1": "M1", "logreg": "M1", "logistic regression": "M1",
    "m2": "M2", "crf": "M2", "linear-chain crf": "M2",
    "m3": "M3", "bilstm-crf": "M3", "bilstm_crf": "M3", "bilstm": "M3",
    "m4": "M4", "bert": "M4", "bert-ft": "M4", "bert fine-tuned": "M4",
    "span": "Span", "span classifier": "Span", "bert span": "Span",
}
SOURCES = ("conll2003", "wnut17", "merged")
TARGETS = ("conll2003", "conllpp", "wikigold", "btc", "wnut17")
DISPLAY = {"conll2003": "CoNLL-03", "conllpp": "CoNLL++", "wikigold": "WikiGold",
           "btc": "BTC", "wnut17": "WNUT-17", "merged": "Merged"}


def canonical_method(value):
    return METHOD_ALIASES.get(value.strip().lower())


def canonical_source(value):
    text = value.strip().lower().replace(" ", "")
    if text in {"merged", "conll2003+wnut17", "wnut17+conll2003"}:
        return "merged"
    return text


def load_rows(paths):
    return [row for path in paths for row in read_csv(path)]


def one_metric(rows, metric, description):
    """Return the value of the single row with this metric; fail otherwise."""
    matches = [row for row in rows if row["metric"] == metric]
    if len(matches) != 1:
        raise ValueError(f"{description}: expected one {metric} row, found {len(matches)}")
    return float(matches[0]["value"])


def f1_bounds(rows, description):
    """Return the lower and upper 95% bootstrap bounds of F1."""
    for lower, upper in (("f1_ci95_lower", "f1_ci95_upper"),
                         ("bootstrap_ci95.f1.0", "bootstrap_ci95.f1.1")):
        if any(row["metric"] == lower for row in rows):
            return one_metric(rows, lower, description), one_metric(rows, upper, description)
    raise ValueError(f"{description}: missing 95% bootstrap F1 bounds")


def save_figure(fig, path):
    """Write the figure to a new PDF, refusing to overwrite an existing file."""
    if path.suffix.lower() != ".pdf":
        raise ValueError("--output must be a vector PDF path")
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        fig.savefig(staged, format="pdf")
        with path.open("xb") as target, staged.open("rb") as source:
            target.write(source.read())
    plt.close(fig)


def draw_main(rows, args):
    """f2: bar chart of F1 per method with intervals and the gain over the previous method."""
    selected = [row for row in rows if row["experiment"] == args.experiment
                and row["dataset"] == args.dataset and row["split"] == args.split
                and row["scope"] == args.scope and row["entity_type"] == "all"
                and row["entity_group"] in {"", "all"} and row["condition"] == args.condition
                and row["label_space"] == args.label_space and row["seed"] == str(args.seed)
                and row["train_source"] == args.train_source]
    values, bounds = [], []
    for name in NAMES:
        group = [row for row in selected if canonical_method(row["method"]) == name]
        values.append(one_metric(group, "f1", name))
        bounds.append(f1_bounds(group, name))
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("F1 values must be ratios between 0 and 1")
    if any(not (0 <= lo <= hi <= 1) for lo, hi in bounds):
        raise ValueError("confidence interval bounds must be ordered ratios")
    style.setup()
    fig, (axis, delta_axis) = plt.subplots(2, 1, figsize=(style.TEXT_WIDTH, 3.2),
                                           gridspec_kw={"height_ratios": (8, 1)}, constrained_layout=True)
    positions = np.arange(len(NAMES))
    axis.bar(positions, np.asarray(values) * 100, color=[style.METHOD[name] for name in NAMES])
    for index, (value, (low, high)) in enumerate(zip(values, bounds, strict=True)):
        axis.vlines(index, low * 100, high * 100, color="#222222", linewidth=1)
        axis.hlines([low * 100, high * 100], index - 0.09, index + 0.09, color="#222222", linewidth=1)
        axis.text(index, min(106, max(value, high) * 100 + 1), f"{value * 100:.1f}",
                  ha="center", va="bottom", fontsize=7)
    axis.set_xticks(positions, NAMES)
    axis.set_ylabel("Entity micro-F1 (%)")
    axis.set_ylim(0, 110)
    delta_axis.axis("off")
    delta_axis.set_xlim(-0.5, len(NAMES) - 0.5)
    delta_axis.text(0, 0.5, "Δ vs prior\n(pp)", ha="center", va="center", fontsize=7)
    for index in range(1, len(NAMES)):
        delta_axis.text(index, 0.5, f"{(values[index]-values[index-1])*100:+.1f}",
                        ha="center", va="center", fontsize=8)
    save_figure(fig, args.output)


def draw_transfer(rows, args):
    """f6a: heat map of BERT F1 for each training source and test dataset."""
    selected = [row for row in rows if row["experiment"] == args.experiment
                and row["split"] == args.split and row["scope"] == args.scope
                and row["entity_type"] == "all" and row["entity_group"] in {"", "all"}
                and row["condition"] == args.condition and row["label_space"] == args.label_space
                and row["seed"] == str(args.seed) and canonical_method(row["method"]) == "M4"]
    matrix = np.empty((len(SOURCES), len(TARGETS)))
    for i, source in enumerate(SOURCES):
        for j, target in enumerate(TARGETS):
            cell = [row for row in selected if canonical_source(row["train_source"]) == source
                    and row["dataset"] == target]
            matrix[i, j] = one_metric(cell, "f1", f"{source} -> {target}")
    if not np.all((matrix >= 0) & (matrix <= 1)):
        raise ValueError("F1 values must be ratios between 0 and 1")
    style.setup()
    fig, axis = plt.subplots(figsize=(style.TEXT_WIDTH, 2.6), constrained_layout=True)
    plotted = axis.imshow(matrix * 100, cmap="cividis", vmin=0, vmax=100, aspect="auto")
    axis.set_xticks(range(len(TARGETS)), [DISPLAY[name] for name in TARGETS])
    axis.set_yticks(range(len(SOURCES)), [DISPLAY[name] for name in SOURCES])
    axis.set_xlabel("Test dataset")
    axis.set_ylabel("Training source")
    axis.grid(False)
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, f"{value * 100:.1f}", ha="center", va="center",
                  color="white" if value < 0.5 else "black", fontsize=8)
    fig.colorbar(plotted, ax=axis, label="Entity micro-F1 (%)")
    save_figure(fig, args.output)


def draw_representations(rows, args):
    """f3b: bar chart of BiLSTM-CRF F1 for the four input representations."""
    selected = [row for row in rows if row["experiment"] == args.experiment
                and row["dataset"] == args.dataset and row["split"] == args.split
                and row["scope"] == args.scope and row["entity_type"] == "all"
                and row["entity_group"] in {"", "all"} and row["label_space"] == args.label_space
                and row["seed"] == str(args.seed) and row["train_source"] == args.train_source
                and canonical_method(row["method"]) == "M3"]
    if len(set(args.conditions)) != 4:
        raise ValueError("four distinct representation conditions are required")
    values, bounds = [], []
    for condition in args.conditions:
        group = [row for row in selected if row["condition"] == condition]
        values.append(one_metric(group, "f1", condition))
        bounds.append(f1_bounds(group, condition))
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("F1 values must be ratios between 0 and 1")
    if any(not (0 <= lo <= hi <= 1) for lo, hi in bounds):
        raise ValueError("confidence interval bounds must be ordered ratios")
    style.setup()
    fig, axis = plt.subplots(figsize=(style.TEXT_WIDTH, 2.7), constrained_layout=True)
    positions = np.arange(4)
    axis.bar(positions, np.asarray(values) * 100, color=style.METHOD["M3"])
    for index, (value, (low, high)) in enumerate(zip(values, bounds, strict=True)):
        axis.vlines(index, low * 100, high * 100, color="#222222", linewidth=1)
        axis.hlines([low * 100, high * 100], index - 0.09, index + 0.09, color="#222222", linewidth=1)
        axis.text(index, min(106, max(value, high) * 100 + 1), f"{value * 100:.1f}",
                  ha="center", va="bottom", fontsize=7)
    axis.set_xticks(positions, ("Random", "GloVe", "GloVe + chars", "Frozen BERT"))
    axis.set_ylabel("Entity micro-F1 (%)")
    axis.set_ylim(0, 110)
    save_figure(fig, args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", choices=("f2", "f3b", "f6a"))
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", help="CSV experiment; defaults to baseline or transfer")
    parser.add_argument("--condition", default="standard")
    parser.add_argument("--split", default="test")
    parser.add_argument("--scope", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-space", help="Sorted type names joined by +")
    parser.add_argument("--dataset", default="conll2003", help="F2 dataset")
    parser.add_argument("--train-source", default="conll2003", help="F2 training source")
    parser.add_argument("--conditions", nargs=4, default=("random", "glove", "glove_char", "bert_frozen"),
                        help="F3(b) CSV condition values, in plotted order")
    args = parser.parse_args()
    args.experiment = args.experiment or {"f2": "baseline", "f3b": "representation", "f6a": "transfer"}[args.figure]
    args.label_space = args.label_space or (
        "LOC+ORG+PER" if args.figure == "f6a" else "LOC+MISC+ORG+PER"
    )
    try:
        if args.output.resolve() in {path.resolve() for path in args.csv}:
            raise ValueError("output must not overwrite an input")
        rows = load_rows(args.csv)
        if args.figure == "f2":
            draw_main(rows, args)
        elif args.figure == "f3b":
            draw_representations(rows, args)
        else:
            draw_transfer(rows, args)
        print(f"Figure: {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
