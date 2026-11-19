"""Draw the B-owned F5 supplement from frozen, checked neural analyses.

The source JSONs are precomputed test results: this script does not run
models, change the protocol, or compare test results to select a model.
It writes a standalone PDF, 300-dpi PNG, and a provenance/caption JSON;
existing files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.style import METHOD, setup

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "output/b_finish_v2/additional_analysis"
OUTPUT = ROOT / "report/figures"
METHODS = ("M3", "M4", "Span")
NAMES = ("BiLSTM-CRF", "BERT", "Span")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load(path, method, *, dataset=None, condition=None):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("method") != method or obj.get("seed") != 0:
        raise ValueError(f"Unexpected method or seed: {path}")
    if dataset and obj.get("dataset") != dataset:
        raise ValueError(f"Unexpected dataset: {path}")
    if condition and obj.get("condition") != condition:
        raise ValueError(f"Unexpected intervention: {path}")
    if obj.get("frozen_manifest_sha256") != FROZEN:
        raise ValueError(f"Manifest mismatch: {path}")
    return obj


FROZEN = "cc20216cd9a430153e9d344eecb3b340bbe3ed46d559c4081ca5885300189d01"


def finite_probability(value):
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid F1 or interval endpoint: {value}")
    return 100.0 * value


def interval(obj, group):
    center = finite_probability(obj[group]["f1"])
    lower, upper = map(finite_probability, obj[group]["f1_ci95"])
    if not lower <= center <= upper:
        raise ValueError("F1 is outside its recorded confidence interval")
    return center, center - lower, upper - center


def draw(output_dir):
    inventory = json.loads((ANALYSIS / "inventory.json").read_text(encoding="utf-8"))
    if inventory.get("frozen_manifest_sha256") != FROZEN:
        raise ValueError("Additional-analysis inventory does not match the frozen manifest")
    if inventory.get("familiarity") != 6 or inventory.get("paired_interventions") != 12:
        raise ValueError("Incomplete analysis inventory")

    data, files = {}, [ANALYSIS / "inventory.json"]
    for method in METHODS:
        paths = {
            "conll": ANALYSIS / f"familiarity__{method}__conll2003.json",
            "conllpp": ANALYSIS / f"familiarity__{method}__conllpp.json",
            "replacement": ANALYSIS / f"paired__{method}__replacement_replaced_v2.json",
        }
        data[method] = {
            "conll": load(paths["conll"], method, dataset="conll2003"),
            "conllpp": load(paths["conllpp"], method, dataset="conllpp"),
            "replacement": load(paths["replacement"], method, condition="replacement_replaced_v2"),
        }
        files.extend(paths.values())
        if (data[method]["conll"]["test_records"] != 3453
                or data[method]["conllpp"]["test_records"] != 1840
                or data[method]["replacement"]["sentence_count"] != 500):
            raise ValueError(f"Unexpected sample counts for {method}")

    outputs = [output_dir / f"b_f5_neural.{ext}" for ext in ("pdf", "png", "metadata.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Refusing to overwrite an existing F5 artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    setup()
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.35), layout="constrained")
    x = np.arange(3, dtype=float)
    for idx, method in enumerate(METHODS):
        color = METHOD[method]
        obj = data[method]["conll"]
        for side, group in ((-1, "seen"), (1, "unseen")):
            center, down, up = interval(obj, group)
            axes[0].bar(x[idx] + side * 0.18, center, width=0.34, color=color,
                        edgecolor="#444444", hatch="///" if group == "unseen" else None,
                        linewidth=0.6, zorder=3)
            axes[0].errorbar(x[idx] + side * 0.18, center, yerr=[[down], [up]], fmt="none",
                             color="#333333", lw=0.8, capsize=2, zorder=4)
            axes[0].text(x[idx] + side * 0.18, center + up + 1.2, f"{center:.1f}",
                         fontsize=6.7, color="#333333", ha="center", va="bottom")
        rep = data[method]["replacement"]
        before, after = map(finite_probability, (rep["first_f1"], rep["second_f1"]))
        axes[1].plot([0, 1], [before, after], marker="o", color=color, label=NAMES[idx], zorder=3)
        for j, y in enumerate((before, after)):
            axes[1].annotate(f"{y:.1f}", (j, y), xytext=(4, (-6, 6, 0)[idx]),
                             textcoords="offset points", va="center", fontsize=7, color=color)
        for date, source in enumerate(("conll", "conllpp")):
            center, down, up = interval(data[method][source], "overall")
            axes[2].errorbar(date, center, yerr=[[down], [up]], fmt="o", color=color,
                             lw=0.8, capsize=2, zorder=4)
        axes[2].plot([0, 1], [interval(data[method][source], "overall")[0]
                              for source in ("conll", "conllpp")], color=color, zorder=3)

    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(facecolor="white", edgecolor="#444444", label="Seen"),
                            Patch(facecolor="white", edgecolor="#444444", hatch="///", label="Unseen")],
                   loc="lower left", ncol=2, fontsize=7, frameon=True,
                   facecolor="white", framealpha=0.95)
    axes[1].legend(loc="lower left", fontsize=7)
    axes[0].set_xticks(x, NAMES, rotation=15, ha="right")
    axes[1].set_xticks([0, 1], ["Original", "Replaced"])
    axes[2].set_xticks([0, 1], ["CoNLL-03", "CoNLL++"])
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.set_ylabel("Entity micro-F1 (%)")
        ax.grid(axis="x", visible=False)
    # Line plots do not imply a zero baseline; explicit ticks show the scale.
    axes[1].set_ylim(80, 100)
    axes[2].set_ylim(75, 100)
    axes[0].set_title("(a) Seen and unseen")
    axes[1].set_title("(b) Name replacement")
    axes[2].set_title("(c) Time shift")
    fig.suptitle("Neural NER: entity familiarity and time shift (seed 0)", fontsize=10)
    fig.savefig(outputs[0], bbox_inches="tight")
    fig.savefig(outputs[1], dpi=300, bbox_inches="tight")
    plt.close(fig)
    caption = ("All values are exact-entity micro-F1 for seed 0. (a) CoNLL-2003 full test, "
               "3,453 sentences, split by case-folded entity surface-form familiarity. "
               "(b) 500 paired original/replaced sentences; values are point estimates; "
               "the paired differences and their intervals are in the metadata. "
               "(c) Full CoNLL-2003 (3,453) and CoNLL++ (1,840) test sets, with different "
               "sentences. Error bars in (a) and (c) are 1,000-sample sentence-bootstrap "
               "95% percentile intervals for each individual score, not seed variability. "
               "Temporal differences are descriptive, not paired tests.")
    metadata = {
        "frozen_manifest_sha256": FROZEN, "caption": caption,
        "sources": {display_path(path): sha256(path) for path in files},
        "outputs": {display_path(path): sha256(path) for path in outputs[:2]},
        "paired_replacement_differences_pp": {
            m: {"delta": 100 * data[m]["replacement"]["second_minus_first"],
                "ci95": [100 * x for x in data[m]["replacement"]["paired_ci95"]]}
            for m in METHODS},
        "limitations": ["Seed 0 only; bootstrap intervals are not between-seed uncertainty.",
                        "Seen/unseen prediction grouping uses name matching irrespective of entity type.",
                        "The 500 paired intervention rows are distinct from the full test populations.",
                        "CoNLL-2003 and CoNLL++ are different test sets, not paired samples."],
    }
    outputs[2].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {outputs[0]}, {outputs[1]} and {outputs[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    draw(args.output_dir.resolve())
