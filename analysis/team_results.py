"""Score all extractors on voting, seen and unseen entities, entity replacement and edited test sets."""

import argparse
import json
from pathlib import Path

import numpy as np

from eval.baseline_statistics import COUNTS, METRICS, bootstrap_counts, compare, metric_values
from eval.ensemble_ner import evaluate as evaluate_ensembles, majority_answers, scoped_predictions
from eval.score_ner import bio_to_spans, read_subset_ids, read_tags_by_id


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "team_analysis.json"
TYPES = ("PER", "LOC", "ORG", "MISC")
BOOTSTRAP = {"resamples": 1000, "seed": 42}
PREFIX = {"M3": "bilstm", "M4": "bert", "Span": "span", "GLiNER": "gliner", "LLM-zero": "qwen7b_k0_plain",
          "LLM": "qwen7b_k10_similar_plain", "LoRA-LLM": "lora1.5b_k0_plain"}
GROUPS = {
    "classical": ["M0", "M1", "M2"],
    "neural": ["M3", "M4", "Span"],
    "cross_crf_bert_llm": ["M2", "M4", "LLM"],
    "cross_lstm_bert_llm": ["M3", "M4", "LLM"],
}


def load_report(name):
    path = ROOT / "results" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Missing {path.relative_to(ROOT)}. Run the command in README.md that writes it.")
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def gold_rows(path):
    if not path.is_file():
        raise ValueError(f"Missing {relative(path)}. Run python -B -m data.prepare --all first.")
    tags = read_tags_by_id(path)
    with path.open(encoding="utf-8-sig") as source:
        rows = {row["sample_id"]: row for line in source if line.strip() for row in (json.loads(line),)}
    for sample_id, row in rows.items():
        if not row.get("tokens") or len(row["tokens"]) != len(tags[sample_id]):
            raise ValueError(f"{relative(path)} / {sample_id}: tokens and labels do not align")
        if {kind for _, _, kind in bio_to_spans(row["tags"])} - set(TYPES):
            raise ValueError(f"{relative(path)} / {sample_id}: expected the four news types")
    return rows


def answer_rows(path, gold, *, subset_ids=None):
    if not path.is_file():
        raise ValueError(f"Missing {relative(path)}. Generate it with the prediction commands in README.md.")
    answers = read_tags_by_id(path)
    key = relative(path)
    if subset_ids is None and answers.keys() != gold.keys():
        raise ValueError(f"{key}: answer IDs must exactly cover the requested gold input")
    scoped = scoped_predictions(key, answers, gold, gold.keys() if subset_ids is None else subset_ids)
    for sample_id, tags in scoped.items():
        if len(tags) != len(gold[sample_id]["tokens"]):
            raise ValueError(f"{key} / {sample_id}: answer and gold token counts differ")
        if {kind for _, _, kind in bio_to_spans(tags)} - set(TYPES):
            raise ValueError(f"{key} / {sample_id}: expected the four news types")
    return scoped


def paths_for(scope, baseline, temporal, robust):
    dataset = "conllpp" if scope == "temporal_conllpp" else "conll2003"
    classic = {}
    if scope in {"original", "temporal_conll2003"}:
        classic = baseline["groups"]["conll2003"]["targets"]["conll2003"]["predictions"]
    elif scope == "temporal_conllpp":
        classic = {method: row["predictions"] for method, row in temporal["datasets"][dataset].items()}
    else:
        experiment = "replacement" if scope.startswith("replacement_") else "perturbations"
        condition = scope.removeprefix("replacement_") if experiment == "replacement" else scope.removesuffix("_v1")
        classic = {method: row["predictions"] for method, row in robust[experiment]["conditions"][condition]["models"].items()}
    paths = {method.upper(): ROOT / path for method, path in classic.items()}

    suffix = ("full" if scope in {"original", "temporal_conll2003"} else "test_300" if scope == "temporal_conllpp"
              else f"{scope}_v2" if scope.startswith("replacement_") else scope)
    for method, prefix in PREFIX.items():
        paths[method] = ROOT / "preds" / dataset / f"{prefix}_{suffix}" / "0.jsonl"
    return paths


def score_counts(counts):
    total = counts.sum(axis=0)
    values = metric_values(total)
    intervals = np.quantile(metric_values(bootstrap_counts(counts, **BOOTSTRAP)), (0.025, 0.975), axis=0)
    return {
        **dict(zip(COUNTS, map(int, total), strict=True)),
        **dict(zip(METRICS, map(float, values), strict=True)),
        "bootstrap_ci95": {metric: intervals[:, index].tolist() for index, metric in enumerate(METRICS)},
    }


def familiarity(rows, answers, known):
    counts = np.zeros((len(rows), 3, len(COUNTS)), dtype=np.int64)
    for index, sample_id in enumerate(sorted(rows)):
        tokens = rows[sample_id]["tokens"]
        gold, predicted = bio_to_spans(rows[sample_id]["tags"]), bio_to_spans(answers[sample_id])
        for group, seen in enumerate((True, False), start=1):
            def matches(span):
                start, end, _ = span
                return (tuple(word.casefold() for word in tokens[start:end]) in known) == seen
            expected, actual = {span for span in gold if matches(span)}, {span for span in predicted if matches(span)}
            counts[index, group] = len(expected & actual), len(actual - expected), len(expected - actual)
    counts[:, 0] = counts[:, 1] + counts[:, 2]
    return {group: score_counts(counts[:, index]) for index, group in enumerate(("overall", "seen", "unseen"))}


def condition_scores(variants, paths):
    """Pair all conditions by sentence ID, including deleted-punctuation variants."""
    conditions = {condition: {"models": {}} for condition in variants}
    ids = sorted(variants["original"])
    methods = paths["original"]
    for method in methods:
        if any(method not in answer_paths for answer_paths in paths.values()):
            raise ValueError(f"{method}: every paired condition must provide an answer path")
        counts = []
        for condition, rows in variants.items():
            if set(rows) != set(ids):
                raise ValueError(f"{condition}: gold IDs differ from the paired original")
            path = paths[condition][method]
            answers = answer_rows(path, rows)
            scores, by_id = compare({key: row["tags"] for key, row in rows.items()}, answers)
            conditions[condition]["models"][method] = {"scores": scores, "predictions": relative(path)}
            counts.append([[by_id[sample_id][key] for key in COUNTS] for sample_id in ids])
        samples = metric_values(bootstrap_counts(np.asarray(counts).transpose(1, 0, 2), **BOOTSTRAP))
        original = conditions["original"]["models"][method]["scores"]
        for index, condition in enumerate(variants):
            row = conditions[condition]["models"][method]
            bounds = np.quantile(samples[:, index], (0.025, 0.975), axis=0)
            differences = np.quantile(100 * (samples[:, index] - samples[:, 0]), (0.025, 0.975), axis=0)
            row["scores"]["bootstrap_ci95"] = {metric: bounds[:, i].tolist() for i, metric in enumerate(METRICS)}
            row["delta_vs_original_pp"] = {metric: 100 * (row["scores"][metric] - original[metric]) for metric in METRICS}
            row["delta_vs_original_ci95_pp"] = {metric: differences[:, i].tolist() for i, metric in enumerate(METRICS)}
    return {"conditions": conditions, "scored_records": len(ids), "scope": "full"}


def build():
    baseline = load_report("m0_m2_baselines")
    temporal = load_report("m0_m2_temporal")
    robust = load_report("m0_m2_robustness")
    rows = gold_rows(ROOT / "data" / "processed" / "conll2003" / "test.jsonl")
    train = gold_rows(ROOT / "data" / "processed" / "conll2003" / "train.jsonl")
    known = {tuple(word.casefold() for word in row["tokens"][start:end])
             for row in train.values() for start, end, _ in bio_to_spans(row["tags"])}
    paths = paths_for("original", baseline, temporal, robust)
    answers = {method: answer_rows(path, rows) for method, path in paths.items()}
    familiar = {method: familiarity(rows, answer, known) for method, answer in answers.items()}
    print(f"Familiarity: {len(familiar)} methods on {len(rows)} complete CoNLL-2003 test sentences", flush=True)

    replacement_rows = {condition: gold_rows(ROOT / "data" / "derived" / "entity_replacement" / "conll2003" /
                                            f"{condition}_v2.jsonl") for condition in ("original", "replaced")}
    replacement_paths = {condition: paths_for(f"replacement_{condition}", baseline, temporal, robust)
                         for condition in replacement_rows}
    replacement = condition_scores(replacement_rows, replacement_paths)
    variants = {"original": rows, **{condition: gold_rows(ROOT / "data" / "derived" / "perturbations" /
                "conll2003" / f"{condition}_v1.jsonl")
                for condition in ("lowercase", "spelling_noise", "no_punctuation")}}
    variant_paths = {condition: paths if condition == "original" else paths_for(f"{condition}_v1", baseline, temporal, robust)
                     for condition in variants}
    perturbations = condition_scores(variants, variant_paths)
    news = {}
    for dataset in ("conll2003", "conllpp"):
        gold = rows if dataset == "conll2003" else gold_rows(ROOT / "data" / "processed" / dataset / "test.jsonl")
        subset_path = ROOT / "data" / "subsets" / dataset / "test_300.txt"
        ids = read_subset_ids(subset_path)
        if len(ids) != 300 or set(ids) - gold.keys():
            raise ValueError(f"{relative(subset_path)}: expected 300 known test IDs")
        news[dataset] = {"models": {}}
        for method, path in paths_for(f"temporal_{dataset}", baseline, temporal, robust).items():
            selected = {sample_id: gold[sample_id] for sample_id in ids}
            prediction = answer_rows(path, gold, subset_ids=ids)
            _, counts = compare({key: row["tags"] for key, row in selected.items()}, prediction)
            news[dataset]["models"][method] = {
                "scores": score_counts(np.array([[counts[key][name] for name in COUNTS] for key in sorted(ids)])),
                "predictions": relative(path),
            }
    print("Replacement, full-test perturbations and both fixed-300 news subsets scored", flush=True)
    ensemble = evaluate_ensembles({key: row["tags"] for key, row in rows.items()}, answers, GROUPS)
    predicted_spans = {method: {sample_id: bio_to_spans(tags) for sample_id, tags in answer.items()}
                       for method, answer in answers.items()}
    gold = {key: row["tags"] for key, row in rows.items()}
    for group in ensemble["groups"].values():
        tags = majority_answers(gold, predicted_spans, group["members"], group["votes_required"])
        _, counts = compare(gold, tags)
        intervals = score_counts(np.array([[counts[key][name] for name in COUNTS] for key in sorted(rows)]))
        group["scores"]["bootstrap_ci95"] = intervals["bootstrap_ci95"]
    return {
        "metric": "Exact entity-level micro precision, recall and F1 on the same sentences and the four CoNLL-2003 types.",
        "entity_types": TYPES, "bootstrap": {**BOOTSTRAP, "unit": "sentence", "interval": "95% percentile"},
        "seen_entity": "An entity whose casefolded words occur as a gold entity of any type in the CoNLL-2003 training split.",
        "ensemble_change": "Difference from the best single member of the group on this test set.",
        "familiarity": {"scored_records": len(rows), "scope": "full", "methods": familiar},
        "replacement": {**replacement, "scope": "paired_500", "version": "v2"},
        "temporal": {"scope": "test_300", "scored_records_per_dataset": 300, "datasets": news},
        "perturbations": {**perturbations, "version": "v1"},
        "ensemble": {"scored_records": len(rows), "scope": "full", **ensemble},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        result = build()
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
