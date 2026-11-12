"""Generate T1 and member A's contributions to T2 and T4 as LaTeX tables."""

import json

from analysis.data_exploration import ROOT


RESULTS = ROOT / "results"
OUTPUT = ROOT / "report" / "tables"


def read(name):
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def table(columns, headers, rows, note):
    return "\n".join([
        "% " + note,
        r"\begingroup\small\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
        *(" & ".join(row) + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
        r"\endgroup",
        "",
    ])


def shortstack(*lines):
    return r"\shortstack[l]{" + r" \\ ".join(lines) + "}"


def dataset_table(report):
    rows = []
    for row in report["datasets"].values():
        splits = row["splits"]
        test = splits["test"]
        counts = "/".join(f"{splits[key]['sentences']:,}" if key in splits else "--" for key in ("train", "validation", "test"))
        types = [name.replace("_", r"\_") for name in test["type_counts"]]
        group = 3 if len(types) >= 8 else 2 if len(types) >= 6 else 4
        labels = shortstack(*(", ".join(types[i:i + group]) for i in range(0, len(types), group)))
        use = row["use"].replace(" and ", " / ")
        words = use.split()
        rows.append((
            shortstack(row["name"], row["genre"]), counts, f"{test['entities']:,}",
            labels, f"{test['mean_sentence_tokens']:.2f}", f"{test['mean_entity_tokens']:.2f}",
            shortstack(" ".join(words[:3]), " ".join(words[3:])) if len(words) > 3 else use,
        ))
    return table(
        "lrrlrrl",
        ("Dataset / text", "Train/Dev/Test", "Entities", "Native types", "Tok./sent.", "Tok./entity", "Use"),
        rows, "T1. Entity counts and mean lengths are measured on test splits. Load booktabs.",
    )


def parameter_table():
    conll, native, controls = (read(name) for name in ("m2_cv_grid", "m2_native_cv", "m2_cv_controls"))
    baselines = read("m0_m2_baselines")["groups"]
    rows = [("M0", "All", "Fixed rule", "Exact, longest match", "--")]
    for key, dataset in (("conll2003", "CoNLL-2003"), ("wnut17", "WNUT-17"), ("mit_restaurant", "MIT Restaurant")):
        model = baselines[key]["m1"]
        values = ",".join(f"{candidate['c']:g}" for candidate in model["selection"]["candidates"])
        rows.append(("M1", dataset, "Source validation F1", rf"$C\in\{{{values}\}}$", rf"$C={model['c']:g}$"))
    datasets = (
        ("CoNLL-2003", conll, "Document-grouped CV"),
        ("WNUT-17", native["datasets"]["wnut17"], "Contiguous block CV"),
        ("MIT Restaurant", native["datasets"]["mit_restaurant"], "Contiguous block CV"),
    )
    for dataset, report, split in datasets:
        values = report["best_parameters"]
        candidates = report["candidates"] + report.get("historical_outside_grid", [])
        c1, c2 = ([candidate[key] for candidate in candidates] for key in ("c1", "c2"))
        ranges = rf"$c_1\in[{min(c1):g},{max(c1):g}],\ c_2\in[{min(c2):g},{max(c2):g}]$"
        rows.append(("M2", dataset, split, shortstack(f"{len(candidates)} pairs", ranges), rf"$({values['c1']:g},{values['c2']:g})$"))
    windows = [controls["higher_mean_window"], *(row["higher_mean_window"] for row in controls["native_windows"]["datasets"].values())]
    rows.append(("M2 window", "Three datasets", "Paired train CV", r"$\pm1$ vs. $\pm2$", r"$" + "/".join(rf"\pm{k}" for k in windows) + "$"))
    return table(
        "lllll", ("Method", "Dataset", "Selection", "Candidates", "Selected / higher"), rows,
        "T2, member A. Five folds for all CV rows. Official test results use +/-1. See the CV reports for every candidate pair. Load booktabs.",
    )


def ensemble_table():
    report = read("m0_m2_ensemble")
    group = report["groups"]["classical"]
    scores = {**report["models"], "Classical majority": group["scores"]}
    rows = [(model, *(f"{100 * result[metric]:.2f}" for metric in ("precision", "recall", "f1")))
            for model, result in scores.items()]
    change = group["f1_delta_vs_best_pp"]
    low, high = group["paired_f1_delta_ci95_pp"]
    return table("lrrr", ("Method", "P (\\%)", "R (\\%)", "F1 (\\%)"), rows,
                 f"T4, member A. CoNLL-2003 test subset ({report['scored_records']} sentences). "
                 f"Majority minus {group['best_constituent']['model']}: {change:+.2f} pp "
                 f"(paired 95% CI: {low:+.2f}, {high:+.2f}). Load booktabs.")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = {"t1": dataset_table(read("data_exploration")), "t2": parameter_table(), "t4_a": ensemble_table()}
    for key, content in tables.items():
        name = f"{key}.tex"
        (OUTPUT / name).write_text(content, encoding="utf-8", newline="\n")
        print(OUTPUT / name)


if __name__ == "__main__":
    main()
