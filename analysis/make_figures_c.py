"""Generate F9: prompt variables on the validation subset and LLM error types with and without annotation rules."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from analysis.data_exploration import ROOT
from analysis.make_figures import EDGE, panel, save
from analysis.style import ERROR, METHOD, TEXT_WIDTH, setup


RESULTS = ROOT / "results" / "c"
COLOR = METHOD["LLM"]
HATCH = "////"
ERROR_LABELS = {
    "boundary": "Boundary", "type": "Type", "boundary_and_type": "Boundary and type",
    "missed": "Missed", "spurious": "Spurious",
}


def grid():
    """Index validation F1 by model size, number of examples, example selection and rule use."""
    scores = {}
    for path in RESULTS.glob("qwen*_conll2003_validation_seed0.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["scope"] != "validation_300":
            continue
        size = "7B" if "7B" in report["model"] else "1.5B"
        scores[size, report["shots"], report["selection"], report["guideline"]] = 100 * report["scores"]["f1"]
    return scores


def paired_bars(ax, labels, first, second, names):
    x = np.arange(len(labels))
    ax.bar(x - 0.19, first, 0.36, color=COLOR, **EDGE, label=names[0])
    ax.bar(x + 0.19, second, 0.36, color=COLOR, hatch=HATCH, **EDGE, label=names[1])
    ax.set_xticks(x, labels)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2, handlelength=1.2, columnspacing=0.8)


def figure_9():
    scores = grid()
    shots = (0, 1, 5, 10)
    fig = plt.figure(figsize=(TEXT_WIDTH, 4.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=(1, 1.05))
    axes = top.subplots(1, 4, sharey=True)

    axes[0].plot(range(4), [scores["7B", k, "similar" if k else None, False] for k in shots],
                 marker="o", color=COLOR)
    axes[0].set_xticks(range(4), shots)
    axes[0].set_xlabel("Examples")
    axes[0].set_ylabel("Validation F1 (%)")

    paired_bars(axes[1], ("1", "5", "10"),
                [scores["7B", k, "random", False] for k in shots[1:]],
                [scores["7B", k, "similar", False] for k in shots[1:]], ("Random", "Similar"))
    axes[1].set_xlabel("Examples")

    paired_bars(axes[2], ("0", "1", "5", "10"),
                [scores["7B", k, "similar" if k else None, False] for k in shots],
                [scores["7B", k, "similar" if k else None, True] for k in shots], ("No rules", "Rules"))
    axes[2].set_xlabel("Examples")

    paired_bars(axes[3], ("0", "10"),
                [scores["1.5B", k, "similar" if k else None, False] for k in (0, 10)],
                [scores["7B", k, "similar" if k else None, False] for k in (0, 10)], ("1.5B", "7B"))
    axes[3].set_xlabel("Examples")
    axes[0].set_ylim(40, 95)
    panel(axes[0], "a")

    ax = bottom.subplots()
    report = json.loads((RESULTS / "error_types.json").read_text(encoding="utf-8"))
    bars = [(dataset, condition) for dataset in ("conll2003", "wnut17") for condition in ("plain", "rules")]
    names = {"conll2003": "CoNLL-03", "wnut17": "WNUT-17"}
    left = np.zeros(len(bars))
    for kind, label in ERROR_LABELS.items():
        values = np.array([report[dataset][condition]["categories"][kind] for dataset, condition in bars])
        ax.barh(range(len(bars)), values, left=left, color=ERROR[kind], **EDGE, label=label)
        for index, value in enumerate(values):
            if value >= 12:
                ax.text(left[index] + value / 2, index, str(value), ha="center", va="center", fontsize=7,
                        color="white" if kind == "spurious" else "#222222")
        left += values
    absent = [report[dataset][condition]["absent_from_sentence"] for dataset, condition in bars]
    ax.barh(range(len(bars)), absent, left=left, color=ERROR["spurious"], hatch=HATCH, **EDGE,
            label="Not in sentence")
    ax.set_yticks(range(len(bars)), [f"{names[dataset]}, {'rules' if condition == 'rules' else 'no rules'}"
                                     for dataset, condition in bars])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Errors on 300 test sentences")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=6, handlelength=1.2, columnspacing=0.8)
    panel(ax, "b")
    save(fig, "f9")


if __name__ == "__main__":
    setup()
    figure_9()
