"""Generate F1, F3(a), F5, F6(b) and F7."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.data_exploration import DATASETS, ROOT
from analysis.style import DATASET, ENTITY, METHOD, TEXT_WIDTH, setup


RESULTS = ROOT / "results"
OUTPUT = ROOT / "report" / "figures"
METHODS = ("m0", "m1", "m2")
TEAM_METHODS = ("M0", "M1", "M2", "M3", "M4", "Span", "GLiNER", "LLM-zero", "LLM", "LoRA-LLM")
TEAM_LABELS = {
    "M0": "M0 Dictionary", "M1": "M1 LogReg", "M2": "M2 CRF", "M3": "M3 BiLSTM-CRF",
    "M4": "M4 BERT", "Span": "Span", "GLiNER": "GLiNER",
    "LLM-zero": "Qwen 0-shot", "LLM": "Qwen 10-shot", "LoRA-LLM": "LoRA Qwen1.5B",
}
SHORT_DATASET = {
    "conll2003": "CoNLL-03", "wnut17": "WNUT-17", "mit_restaurant": "MIT Rest.",
    "conllpp": "CoNLL++", "wikigold": "WikiGold", "btc": "BTC", "merged": "Merged",
}
PERTURBATIONS = {"lowercase": "Lowercase", "spelling_noise": "5% spelling", "no_punctuation": "No punctuation"}
NATIVE_COLOR = {"person": ENTITY["PER"], "location": ENTITY["LOC"],
                "corporation": ENTITY["ORG"], "group": "#CC6677", "product": "#AA4499", "creative-work": "#88CCEE",
                "Amenity": "#44AA99", "Cuisine": "#CC6677", "Dish": "#AA4499", "Hours": "#DDCC77",
                "Location": ENTITY["LOC"], "Price": "#117733", "Rating": "#332288", "Restaurant_Name": ENTITY["ORG"]}


def read(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def panel(ax, letter):
    ax.text(0, 1.02, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")


def interval(ax, x, low, high):
    ax.errorbar(x, (low + high) / 2, yerr=(high - low) / 2, fmt="none",
                ecolor="#444444", linewidth=0.7, capsize=2, zorder=3)


EDGE = {"edgecolor": "#444444", "linewidth": 0.5}


def save(figure, name):
    for extension in ("pdf", "png"):
        figure.savefig(OUTPUT / f"{name}.{extension}")
    plt.close(figure)
    print(OUTPUT / f"{name}.pdf")


def figure_1():
    report = read("data_exploration")
    keys = list(report["datasets"])
    test = {key: report["datasets"][key]["splits"]["test"] for key in keys}
    fig = plt.figure(figsize=(TEXT_WIDTH, 6.9), layout="constrained")
    grid = fig.add_gridspec(3, 6, height_ratios=(1, 1, 1.45))
    for i, dataset in enumerate(keys):
        column = (i % 3) * 2
        ax = fig.add_subplot(grid[i // 3, column:column + 2])
        counts = test[dataset]["type_counts"]
        kinds = sorted(counts, key=counts.get, reverse=True)
        values = [100 * counts[kind] / test[dataset]["entities"] for kind in kinds]
        ax.barh(range(len(kinds)), values, color=[ENTITY.get(kind, NATIVE_COLOR.get(kind, "#BBBBBB")) for kind in kinds])
        ax.set_yticks(range(len(kinds)), [kind.replace("_", " ").replace("creative-work", "creative work") for kind in kinds])
        ax.invert_yaxis()
        ax.set_xticks((0, 25, 50, 75), ("0", "25", "50", "75"))
        ax.set_xlim(0, 75)
        ax.set_xlabel(f"{SHORT_DATASET[dataset]} (%)")
        ax.grid(axis="y", visible=False)
        if i == 0:
            panel(ax, "a")

    ax = fig.add_subplot(grid[2, :3])
    labels = [SHORT_DATASET[key] for key in keys]
    y = np.arange(len(keys))
    ax.scatter([100 * test[k]["initial_uppercase_fraction"] for k in keys], y, marker="o", facecolors="none",
               edgecolors="#333333", label="Initial uppercase", zorder=3)
    ax.scatter([100 * test[k]["all_lowercase_fraction"] for k in keys], y, marker="x",
               color="#333333", label="All lowercase", zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(-4, 104)
    ax.set_xlabel("Share of entity mentions (%)")
    ax.legend(loc="center right", bbox_to_anchor=(0.98, 0.58))
    panel(ax, "b")

    ax = fig.add_subplot(grid[2, 3:])
    values = np.array([[100 * report["test_name_overlap"][left][right]["jaccard"] for right in keys] for left in keys])
    off_diag = np.ma.array(values, mask=np.eye(len(keys), dtype=bool))
    palette = matplotlib.colormaps["cividis"].with_extremes(bad="#E4E4E4")
    chart = ax.imshow(off_diag, cmap=palette, vmin=0, vmax=off_diag.max(), interpolation="nearest", aspect="equal")
    for row, col in np.ndindex(values.shape):
        color = "#222222" if row == col or values[row, col] / chart.norm.vmax > 0.58 else "white"
        ax.text(col, row, f"{values[row, col]:.1f}" if row != col else "100", color=color, ha="center", va="center", fontsize=7)
    ax.set_xticks(y, labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(y, labels)
    ax.grid(False)
    fig.colorbar(chart, ax=ax, shrink=0.84, label="Name overlap (%)")
    panel(ax, "c")
    save(fig, "f1")


def figure_3a():
    report = read("m2_feature_ablation")
    groups = list(report["feature_groups"])
    kinds = list(ENTITY)
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 2.65), layout="constrained")
    for i, kind in enumerate(kinds):
        values = [report["summary"][f"without_{group}"]["mean_per_type_f1_delta_vs_full_pp"][kind] for group in groups]
        ax.bar(x + (i - 1.5) * 0.19, values, 0.18, label=kind, color=ENTITY[kind])
    ax.set_xticks(x, groups)
    ax.set_xlabel("Removed feature group")
    ax.set_ylabel("Mean F1 change (pp)")
    ax.axhline(0, color="#333333", linewidth=0.7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=4)
    panel(ax, "a")
    save(fig, "f3a")


def figure_5():
    team = read("team_analysis")
    extra = read("m2_extra_features")["conditions"]
    replacement = team["replacement"]["conditions"]
    temporal = team["temporal"]["datasets"]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 7.4), layout="constrained")
    first, second, third, fourth = axes.flat
    for index, method in enumerate(TEAM_METHODS):
        color = METHOD["LLM" if method == "LLM-zero" else method]
        familiarity = team["familiarity"]["methods"][method]
        for condition_index, group in enumerate(("seen", "unseen")):
            location = index + (condition_index - 0.5) * 0.36
            scores = familiarity[group]
            first.barh(location, 100 * scores["f1"], 0.32, color=color,
                       hatch="////" if condition_index else None, **EDGE)
            horizontal_interval(first, location, *(100 * value for value in scores["bootstrap_ci95"]["f1"]))
        for condition_index, condition in enumerate(("original", "replaced")):
            location = index + (condition_index - 0.5) * 0.36
            scores = replacement[condition]["models"][method]["scores"]
            second.barh(location, 100 * scores["f1"], 0.32, color=color,
                        hatch="////" if condition_index else None, **EDGE)
            horizontal_interval(second, location, *(100 * value for value in scores["bootstrap_ci95"]["f1"]))

        news = [temporal[dataset]["models"][method]["scores"] for dataset in ("conll2003", "conllpp")]
        values = [100 * scores["f1"] for scores in news]
        third.plot(values, [index, index], color=color if method != "M0" else "#777777", linewidth=1.2)
        for corpus_index, scores in enumerate(news):
            horizontal_interval(third, index, *(100 * value for value in scores["bootstrap_ci95"]["f1"]))
            third.scatter(values[corpus_index], index, marker="o" if corpus_index == 0 else "s", s=24,
                          facecolor=color if corpus_index == 0 else "white", edgecolor="#444444", linewidth=0.65, zorder=4)

    for ax, title in ((first, "(a) Familiarity: full CoNLL-2003"),
                      (second, "(b) Name replacement: 500 pairs"),
                      (third, "(c) News transfer: fixed 300 per corpus")):
        ax.set_yticks(range(len(TEAM_METHODS)), [TEAM_LABELS[method] for method in TEAM_METHODS], fontsize=8)
        ax.set_ylim(len(TEAM_METHODS) - 0.4, -0.6)
        ax.set_xlim(-2, 102)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.set_xlabel("Entity-level F1 (%)")
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9, loc="left", pad=30)
    solid = Patch(facecolor="#DDDDDD", **EDGE)
    hatched = Patch(facecolor="#DDDDDD", hatch="////", **EDGE)
    first.legend((solid, hatched), ("Seen", "Unseen"), loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)
    second.legend((solid, hatched), ("Original", "Replaced"), loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)
    third.legend((Line2D([0], [0], marker="o", linestyle="", color="#777777"),
                  Line2D([0], [0], marker="s", linestyle="", markerfacecolor="white", color="#444444")),
                 ("CoNLL-2003", "CoNLL++"), loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2)

    conditions = ["gazetteer", "glove_100", "glove_500", "glove_1000"]
    x = np.arange(len(conditions))
    for j, (dataset, scope, label) in enumerate((("conll2003", "unseen", "CoNLL-03 unseen"), ("conllpp", "overall", "CoNLL++ all"))):
        values = [extra[condition][dataset]["f1_delta_vs_baseline_pp"][scope] for condition in conditions]
        fourth.bar(x + (j - 0.5) * 0.36, values, 0.34, color=METHOD["M2"], hatch="////" if j else None, **EDGE, label=label)
    fourth.set_xticks(x, ("Gazetteer", "$k=100$", "$k=500$", "$k=1000$"))
    fourth.set_ylabel("F1 change (pp)")
    fourth.axhline(0, color="#333333", linewidth=0.7)
    fourth.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=1)
    fourth.set_title("(d) CRF extra-feature gains", fontsize=9, loc="left", pad=41)
    fig.supxlabel("Error bars: 95% intervals from 1,000 sentence bootstrap resamples.", fontsize=8)
    save(fig, "f5")


def horizontal_interval(ax, y, low, high):
    ax.errorbar((low + high) / 2, y, xerr=(high - low) / 2, fmt="none",
                ecolor="#333333", linewidth=0.7, capsize=2, zorder=3)


def figure_6b():
    report = read("m2_transfer_overlap")
    source = "conll2003"
    rows = [row for row in report["points"] if row["source"] == source and not row["target_is_source_domain"]]
    markers = {"PER": "o", "LOC": "s", "ORG": "^"}
    targets = list(dict.fromkeys(row["target"] for row in rows))
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.2), layout="constrained")
    for row in rows:
        ax.scatter(100 * row["overlap"], row["f1_loss_pp"], marker=markers[row["entity_type"]],
                   color=DATASET[row["target"]], s=36, edgecolor="#555555", linewidth=0.45, zorder=3)
    fit = report["by_source_non_source_targets"][source]
    x = np.array([min(100 * row["overlap"] for row in rows), max(100 * row["overlap"] for row in rows)])
    ax.plot(x, fit["ols_intercept_pp"] + x * fit["ols_slope_f1_loss_pp_per_overlap_pp"], "--", color="#333333")
    ax.text(0.98, 0.72, f"$r={fit['pearson_r']:.2f}$", transform=ax.transAxes, ha="right", va="top")
    ax.set_ylim(bottom=-3)
    ax.set_xlabel("Target entity mentions seen in CoNLL-03 training (%)")
    ax.set_ylabel("F1 loss vs. CoNLL-03 test (pp)")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=DATASET[key], markeredgecolor="#555555",
                      markeredgewidth=0.45, label=SHORT_DATASET[key]) for key in targets]
    type_handles = [Line2D([0], [0], marker=marker, linestyle="", color="#888888", markeredgecolor="#555555",
                           markeredgewidth=0.45, label=kind) for kind, marker in markers.items()]
    legend = ax.legend(handles=handles, loc="upper right", title="Target", ncols=2)
    ax.add_artist(legend)
    ax.legend(handles=type_handles, loc="upper right", bbox_to_anchor=(1, 0.62), title="Entity type", ncols=3)
    panel(ax, "b")
    save(fig, "f6b")


def figure_7():
    g2 = read("m2_g2_transfer")["rows"]
    conditions = read("team_analysis")["perturbations"]["conditions"]
    fig = plt.figure(figsize=(TEXT_WIDTH, 6.0), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=(1, 2.2))
    first = fig.add_subplot(grid[0, :])
    targets = (("conll2003", "conll2003"), ("wnut17", "wnut17"), ("mit_restaurant", "mit_restaurant"), ("conll2003", "btc"))
    deltas = [g2[source]["targets"][target]["without_G2_minus_full_f1_pp"] for source, target in targets]
    palette = matplotlib.colormaps["PuOr"]
    first.barh(range(len(targets)), deltas, color=[palette(0.76 if value < 0 else 0.22) for value in deltas])
    first.set_yticks(range(len(targets)), [SHORT_DATASET[target] for _, target in targets])
    first.invert_yaxis()
    first.axvline(0, color="#333333", linewidth=0.7)
    first.set_xlabel("F1 change without G2 (pp)")
    first.grid(axis="y", visible=False)
    first.set_title("(a) CRF spelling-feature ablation", fontsize=9, loc="left")

    facets = []
    bounds = [value for condition in PERTURBATIONS for method in TEAM_METHODS
              for value in conditions[condition]["models"][method]["delta_vs_original_ci95_pp"]["f1"]]
    low = min(0, min(bounds)) - 3
    high = max(0, max(bounds)) + 2
    for column, (condition, title) in enumerate(PERTURBATIONS.items()):
        ax = fig.add_subplot(grid[1, column], sharex=facets[0] if facets else None,
                             sharey=facets[0] if facets else None)
        facets.append(ax)
        for index, method in enumerate(TEAM_METHODS):
            row = conditions[condition]["models"][method]
            color = METHOD["LLM" if method == "LLM-zero" else method]
            ax.barh(index, row["delta_vs_original_pp"]["f1"], 0.62, color=color, **EDGE)
            horizontal_interval(ax, index, *row["delta_vs_original_ci95_pp"]["f1"])
        ax.set_ylim(len(TEAM_METHODS) - 0.4, -0.6)
        ax.set_xlim(low, high)
        ax.set_yticks(range(len(TEAM_METHODS)), [TEAM_LABELS[method] for method in TEAM_METHODS], fontsize=8)
        ax.tick_params(axis="y", labelleft=column == 0)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color="#333333", linewidth=0.7)
        ax.set_title(f"(b) {title}" if column == 0 else title, fontsize=9, loc="left")
        ax.set_xlabel("F1 change (pp)")
    fig.supxlabel("Error bars: paired 95% intervals from 1,000 sentence bootstrap resamples.", fontsize=8)
    save(fig, "f7")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    setup()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for plot in (figure_1, figure_3a, figure_5, figure_6b, figure_7):
        plot()


if __name__ == "__main__":
    main()
