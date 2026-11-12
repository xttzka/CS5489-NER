"""Generate F1, F3(a), F5, F6(b), F7 and speed–accuracy frontiers."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

from analysis.data_exploration import DATASETS, ROOT
from analysis.style import DATASET, ENTITY, METHOD, TEXT_WIDTH, setup
from eval.result_tables import read_csv, validate_rows


RESULTS = ROOT / "results"
OUTPUT = ROOT / "report" / "figures"
METHODS = ("m0", "m1", "m2")
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
    temporal = read("m0_m2_temporal")["datasets"]
    extra = read("m2_extra_features")["conditions"]
    replacement = read("m0_m2_robustness")["replacement"]["conditions"]
    fig, axes = plt.subplots(2, 2, figsize=(TEXT_WIDTH, 4.6), layout="constrained")
    first, second, third, fourth = axes.flat
    for i, method in enumerate(METHODS):
        row = temporal["conll2003"][method]
        first.bar(i - 0.17, 100 * row["seen"]["f1"], 0.32, color=METHOD[method.upper()], **EDGE,
                  label="Seen" if i == 0 else None)
        first.bar(i + 0.17, 100 * row["unseen"]["f1"], 0.32, color=METHOD[method.upper()], hatch="////", **EDGE,
                  label="Unseen" if i == 0 else None)
    first.set_xticks(range(len(METHODS)), [method.upper() for method in METHODS])
    first.set_ylabel("F1 (%)")
    first.set_ylim(0, 105)
    first.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)

    for i, method in enumerate(METHODS):
        for j, condition in enumerate(("original", "replaced")):
            x = i + (j - 0.5) * 0.34
            scores = replacement[condition]["models"][method]["scores"]
            second.bar(x, 100 * scores["f1"], 0.32,
                       color=METHOD[method.upper()], hatch="////" if j else None, **EDGE,
                       label=condition.capitalize() if i == 0 else None)
            interval(second, x, *(100 * value for value in scores["bootstrap_ci95"]["f1"]))
    second.set_xticks(range(len(METHODS)), [method.upper() for method in METHODS])
    second.set_ylim(0, 105)
    second.set_ylabel("F1 (%)")
    second.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)

    for method, marker in zip(METHODS, ("o", "s", "^"), strict=True):
        third.plot([0, 1], [100 * temporal[dataset][method]["overall"]["f1"] for dataset in ("conll2003", "conllpp")],
                   marker=marker, color=METHOD[method.upper()] if method != "m0" else "#777777", label=method.upper())
    third.set_xticks([0, 1], ("CoNLL-2003", "CoNLL++"))
    third.set_xlim(-0.08, 1.08)
    third.set_ylabel("F1 (%)")
    third.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=3)

    conditions = ["gazetteer", "glove_100", "glove_500", "glove_1000"]
    x = np.arange(len(conditions))
    for j, (dataset, scope, label) in enumerate((("conll2003", "unseen", "CoNLL-03 unseen"), ("conllpp", "overall", "CoNLL++ all"))):
        values = [extra[condition][dataset]["f1_delta_vs_baseline_pp"][scope] for condition in conditions]
        fourth.bar(x + (j - 0.5) * 0.36, values, 0.34, color=METHOD["M2"], hatch="////" if j == 0 else None, **EDGE, label=label)
    fourth.set_xticks(x, ("Gazetteer", "$k=100$", "$k=500$", "$k=1000$"))
    fourth.set_ylabel("F1 change (pp)")
    fourth.axhline(0, color="#333333", linewidth=0.7)
    fourth.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=2)
    for ax, letter in zip(axes.flat, "abcd", strict=True):
        panel(ax, letter)
    save(fig, "f5")


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
    conditions = read("m0_m2_robustness")["perturbations"]["conditions"]
    fig, (first, second) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.95), layout="constrained")
    targets = (("conll2003", "conll2003"), ("wnut17", "wnut17"), ("mit_restaurant", "mit_restaurant"), ("conll2003", "btc"))
    deltas = [g2[source]["targets"][target]["without_G2_minus_full_f1_pp"] for source, target in targets]
    palette = matplotlib.colormaps["PuOr"]
    first.barh(range(len(targets)), deltas, color=[palette(0.76 if value < 0 else 0.22) for value in deltas])
    first.set_yticks(range(len(targets)), [SHORT_DATASET[target] for _, target in targets])
    first.invert_yaxis()
    first.axvline(0, color="#333333", linewidth=0.7)
    first.set_xlabel("F1 change without G2 (pp)")
    first.grid(axis="y", visible=False)
    panel(first, "a")

    x = np.arange(len(PERTURBATIONS))
    for i, method in enumerate(METHODS):
        rows = [conditions[key]["models"][method] for key in PERTURBATIONS]
        values = [row["delta_vs_original_pp"]["f1"] for row in rows]
        locations = x + (i - (len(METHODS) - 1) / 2) * 0.23
        second.bar(locations, values, 0.22, color=METHOD[method.upper()], label=method.upper())
        for location, row in zip(locations, rows, strict=True):
            interval(second, location, *row["delta_vs_original_ci95_pp"]["f1"])
    second.set_xticks(x, list(PERTURBATIONS.values()), rotation=15, ha="right")
    second.set_ylabel("F1 change (pp)")
    second.axhline(0, color="#333333", linewidth=0.7)
    second.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncols=3)
    panel(second, "b")
    save(fig, "f7")


def figure_10_a():
    timing = read("m0_m2_timing")["datasets"]
    baseline = read("m0_m2_baselines")["groups"]
    panels = {}
    for dataset, record in timing.items():
        panels[dataset] = {}
        for method, data in record["models"].items():
            ci = baseline[dataset]["targets"][dataset]["full"]["models"][method]["bootstrap_ci95"]["f1"]
            panels[dataset][method.upper()] = data["inference_sentences_per_second"], data["full_test_f1"], ci
    plot_frontiers(panels, "f10_a")


def team_costs(path, scope):
    """Join test F1 and timing only when member, model, training source and labels agree."""
    metrics = defaultdict(dict)
    for row in validate_rows(read_csv(path)):
        if row["split"] != "test" or row["scope"] != scope or row["condition"] != "standard":
            continue
        if row["dataset"] not in {"conll2003", "wnut17", "mit_restaurant"}:
            continue
        if row["entity_type"] != "all" or row["entity_group"] != "all":
            continue
        pair = row["experiment"], row["metric"]
        if pair not in {("baseline", "f1"), ("timing", "inference_sentences_per_second")}:
            continue
        key = tuple(row[field] for field in ("dataset", "member", "method", "train_source", "label_space"))
        metric = row["metric"]
        by_seed = metrics[key].setdefault(metric, {})
        value = float(row["value"])
        unit = "ratio" if metric == "f1" else "sentences_per_second"
        if row["unit"] != unit or (not 0 <= value <= 1 if metric == "f1" else value <= 0):
            raise ValueError(f"{key}: invalid {metric} or unit")
        if row["seed"] in by_seed:
            raise ValueError(f"{key}: several standard {metric} rows for seed {row['seed']!r}")
        by_seed[row["seed"]] = value

    panels = defaultdict(dict)
    panel_labels = {}
    for (dataset, member, method, source, labels), values in metrics.items():
        if len(values) != 2:
            print(f"Missing paired score or timing: {member}/{method}/{dataset}/{source or 'zero-shot'}")
            continue
        if not labels or panel_labels.setdefault(dataset, labels) != labels:
            raise ValueError(f"{dataset}: frontier points must share the same label space")
        if method in panels[dataset]:
            raise ValueError(f"{dataset}/{method}: use distinct method names for separate training sources or members")
        panels[dataset][method] = median(values["inference_sentences_per_second"].values()), mean(values["f1"].values()), None
    if not panels:
        raise ValueError(f"no matching standard F1 and throughput rows for {scope}")
    return panels


def plot_frontiers(panels, name):
    fig, axes = plt.subplots(1, len(panels), figsize=(TEXT_WIDTH, 2.8), sharex=True, sharey=True, squeeze=False, layout="constrained")
    speeds = [speed for methods in panels.values() for speed, _, _ in methods.values()]
    tick_positions = (1, 2, 5) if max(speeds) / min(speeds) < 200 else (1,)
    labels = {}
    for ax, (dataset, methods) in zip(axes[0], panels.items(), strict=True):
        points = []
        for method, (speed, score, ci) in methods.items():
            f1 = 100 * score
            if ci is not None:
                interval(ax, speed, *(100 * value for value in ci))
            dot = ax.scatter(speed, f1, s=24, edgecolors="#444444", color=METHOD.get(method, "#777777"), zorder=3)
            labels.setdefault(method, dot)
            points.append((speed, f1))
        frontier = [point for point in points if not any(
            other[0] >= point[0] and other[1] >= point[1] and other != point for other in points
        )]
        if len(frontier) > 1:
            ax.plot(*zip(*sorted(frontier)), color="#777777", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_ylim(-4, 104)
        ax.xaxis.set_major_locator(LogLocator(subs=tick_positions, numticks=5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(SHORT_DATASET[dataset])
    axes[0][0].set_ylabel("F1 (%)")
    fig.legend(labels.values(), labels.keys(), loc="outside upper center", ncols=min(len(labels), 6))
    fig.supxlabel("Inference sentences/s (log scale)")
    save(fig, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-results", type=Path, help="Merged team CSV for F10")
    parser.add_argument("--scope", choices=("full", "test_300"), default="full", help="Evaluation scope for team F10")
    args = parser.parse_args()
    setup()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.team_results:
        try:
            plot_frontiers(team_costs(args.team_results, args.scope), f"f10_team_{args.scope}")
        except (OSError, ValueError) as error:
            parser.exit(2, f"Error: {error}\n")
        return
    for plot in (figure_1, figure_3a, figure_5, figure_6b, figure_7, figure_10_a):
        plot()


if __name__ == "__main__":
    main()
