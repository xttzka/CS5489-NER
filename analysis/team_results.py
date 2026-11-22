"""Score the delivered answers for team voting, familiarity, replacement and robustness."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from eval.baseline_statistics import COUNTS, METRICS, bootstrap_counts, compare, metric_values
from eval.ensemble_ner import evaluate as evaluate_ensembles, scoped_predictions
from eval.score_ner import bio_to_spans, read_subset_ids, read_tags_by_id


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "team_analysis.json"
B_REVISION = "dc7c624"
MAIN_REVISION = "b74eaa8"
TYPES = ("PER", "LOC", "ORG", "MISC")
BOOTSTRAP = {"resamples": 1000, "seed": 42}
METHODS = {
    "M0": {"label": "Dictionary", "paradigm": "classic"},
    "M1": {"label": "Logistic regression", "paradigm": "classic"},
    "M2": {"label": "CRF", "paradigm": "classic"},
    "M3": {"label": "BiLSTM-CRF", "paradigm": "neural", "seed": 0},
    "M4": {"label": "BERT", "paradigm": "neural", "seed": 0},
    "Span": {"label": "Span classifier", "paradigm": "neural", "seed": 0},
    "GLiNER": {"label": "GLiNER", "paradigm": "zero-shot"},
    "LLM-zero": {"label": "Qwen7B zero-shot", "paradigm": "LLM"},
    "LLM": {"label": "Qwen7B 10-shot", "paradigm": "LLM"},
    "LoRA-LLM": {"label": "LoRA-Qwen1.5B", "paradigm": "LLM", "seed": 0},
}
NEURAL = {"M3": "bilstm_crf", "M4": "bert", "Span": "span_bert"}
C_PREFIX = {"GLiNER": "gliner", "LLM-zero": "qwen7b_k0_plain", "LLM": "qwen7b_k10_similar_plain",
            "LoRA-LLM": "lora1.5b_k0_plain"}
GROUPS = {
    "classical": ["M0", "M1", "M2"],
    "neural": ["M3", "M4", "Span"],
    "cross_crf_bert_llm": ["M2", "M4", "LLM"],
    "cross_lstm_bert_llm": ["M3", "M4", "LLM"],
}


def load_report(name):
    path = ROOT / "results" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Missing {path.relative_to(ROOT)}. Run its documented experiment before team scoring.")
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def remember(path, inputs, *, records=None):
    content = path.read_bytes()
    metadata = {"sha256": sha256(content).hexdigest(),
                "lf_sha256": sha256(content.replace(b"\r\n", b"\n")).hexdigest()}
    if records is not None:
        metadata["records"] = records
    inputs[relative(path)] = metadata


def matches_digest(path, expected):
    content = path.read_bytes()
    return expected in {sha256(content).hexdigest(), sha256(content.replace(b"\r\n", b"\n")).hexdigest()}


def restore_b_file(path):
    """Retrieve one answer from the pinned delivery without changing the working branch."""
    result = subprocess.run(["git", "show", f"{B_REVISION}:{relative(path)}"], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode:
        raise ValueError(f"Missing {relative(path)}. Run git fetch origin to obtain B's delivery {B_REVISION}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        staged.write_bytes(result.stdout)
        staged.replace(path)


def b_manifest(inputs):
    path = ROOT / "preds" / "b" / "manifest.json"
    if not path.is_file():
        restore_b_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    remember(path, inputs)
    manifest = {row["prediction_file"]: row for row in report["predictions"]}
    if len(manifest) != report["prediction_count"]:
        raise ValueError("B's manifest has repeated answer paths or an inconsistent prediction count")
    return manifest


def gold_rows(path, inputs):
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
    remember(path, inputs, records=len(rows))
    return rows


def answer_rows(path, gold, inputs, manifest, *, subset_ids=None):
    if not path.is_file() and relative(path).startswith("preds/b/"):
        restore_b_file(path)
    if not path.is_file():
        raise ValueError(f"Missing {relative(path)}. Generate the classic answers with run_baselines, "
                         "m2_pos_transfer, run_temporal_baselines and run_robustness; obtain C's tracked predictions with git fetch origin.")
    answers = read_tags_by_id(path)
    key = relative(path)
    if key.startswith("preds/b/"):
        expected = manifest[key]
        if not matches_digest(path, expected["prediction_sha256"]):
            raise ValueError(f"{key}: does not match B's pinned prediction checksum")
        if len(answers) != expected["sentences"]:
            raise ValueError(f"{key}: sentence count differs from B's manifest")
        input_path = ROOT / expected["input"]
        if not matches_digest(input_path, expected["input_sha256"]):
            raise ValueError(f"{key}: evaluation input differs from B's manifest checksum")
        if expected["subset_ids"]:
            subset_path = ROOT / expected["subset_ids"]
            if not matches_digest(subset_path, expected["subset_sha256"]):
                raise ValueError(f"{key}: subset IDs differ from B's manifest checksum")
    if subset_ids is None and answers.keys() != gold.keys():
        raise ValueError(f"{key}: answer IDs must exactly cover the requested gold input")
    scoped = scoped_predictions(key, answers, gold, gold.keys() if subset_ids is None else subset_ids)
    for sample_id, tags in scoped.items():
        if len(tags) != len(gold[sample_id]["tokens"]):
            raise ValueError(f"{key} / {sample_id}: answer and gold token counts differ")
        if {kind for _, _, kind in bio_to_spans(tags)} - set(TYPES):
            raise ValueError(f"{key} / {sample_id}: expected the four news types")
    remember(path, inputs, records=len(answers))
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

    for method, name in NEURAL.items():
        if scope in {"original", "temporal_conll2003"}:
            folder = "domain__train-conll2003__full__standard"
        elif scope == "temporal_conllpp":
            folder = "generalization300__train-conll2003__test_300__standard"
        else:
            folder = f"robustness__train-conll2003__full__{scope}_v2" if scope.startswith("replacement_") else f"robustness__train-conll2003__full__{scope}"
        paths[method] = ROOT / "preds" / "b" / dataset / name / folder / "seed0.jsonl"
    for method, prefix in C_PREFIX.items():
        suffix = "full" if scope in {"original", "temporal_conll2003"} else "test_300" if scope == "temporal_conllpp" else f"{scope}_v2" if scope.startswith("replacement_") else scope
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


def condition_scores(variants, paths, inputs, manifest):
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
            answers = answer_rows(path, rows, inputs, manifest)
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


def method_metadata(baseline, inputs):
    methods = {method: dict(details) for method, details in METHODS.items()}
    for method in ("M0", "M1", "M2"):
        config = baseline["groups"]["conll2003"][method.lower()]
        methods[method]["selection_source"] = f"results/m0_m2_baselines.json#/groups/conll2003/{method.lower()}"
        methods[method]["training_source"] = "conll2003:train"
        if method == "M1":
            methods[method]["parameters"] = {key: config[key] for key in ("c", "solver", "tol", "max_iterations")}
        elif method == "M2":
            methods[method]["parameters"] = config["parameters"]
    selection_path = ROOT / "results" / "b" / "hyperparameter_selection.json"
    selected = json.loads(selection_path.read_text(encoding="utf-8"))["selected"]
    remember(selection_path, inputs)
    for method in NEURAL:
        methods[method].update(
            selection_source=f"results/b/hyperparameter_selection.json#/selected/{method}/conll2003",
            parameters=selected[method]["conll2003"]["settings"], training_source="conll2003:train",
            prediction_revision=B_REVISION,
        )
    for method, name in (
        ("GLiNER", "gliner_conll2003_test_full"),
        ("LLM-zero", "qwen7b_k0_plain_validation_300_conll2003_validation_seed0"),
        ("LLM", "qwen7b_k10_similar_plain_validation_300_conll2003_validation_seed0"),
        ("LoRA-LLM", "lora1.5b_k0_plain_full_conll2003_test_seed0"),
    ):
        path = ROOT / "results" / "c" / f"{name}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        remember(path, inputs)
        methods[method]["configuration_source"] = relative(path)
        methods[method]["parameters"] = {key: report[key] for key in (
            "model", "threshold", "shots", "selection", "guideline", "max_new_tokens", "adapter",
        ) if key in report}
    methods["LLM"]["selection_source"] = methods["LLM"]["configuration_source"]
    methods["GLiNER"]["selection_source"] = "Fixed delivered zero-shot configuration and threshold"
    methods["LLM-zero"]["selection_source"] = "Prespecified zero-shot comparator"
    methods["LoRA-LLM"]["training_source"] = "10% of conll2003:train"
    methods["LoRA-LLM"]["prediction_revision"] = MAIN_REVISION
    original = load_report("c/lora1.5b_k0_plain_full_conll2003_test_seed0")
    methods["LoRA-LLM"]["edited_test_sources"] = {}
    for scope in ("replacement_original_v2", "replacement_replaced_v2", "lowercase_v1",
                  "spelling_noise_v1", "no_punctuation_v1"):
        path = ROOT / "results" / "c" / f"lora1.5b_k0_plain_{scope}_conll2003_test_seed0.json"
        delivered = json.loads(path.read_text(encoding="utf-8"))
        for key in ("method", "model", "adapter", "dataset", "split", "label_scheme", "shots", "seed",
                    "guideline", "type_descriptions", "max_new_tokens"):
            if delivered[key] != original[key]:
                raise ValueError(f"{relative(path)}: {key} differs from the delivered full-original configuration")
        if delivered["scope"] != scope:
            raise ValueError(f"{relative(path)}: unexpected edited-test scope")
        remember(path, inputs)
        methods["LoRA-LLM"]["edited_test_sources"][scope] = relative(path)
    return methods


def build():
    inputs = {}
    manifest = b_manifest(inputs)
    baseline = load_report("m0_m2_baselines")
    temporal = load_report("m0_m2_temporal")
    robust = load_report("m0_m2_robustness")
    for name in ("m0_m2_baselines", "m0_m2_temporal", "m0_m2_robustness"):
        remember(ROOT / "results" / f"{name}.json", inputs)
    methods = method_metadata(baseline, inputs)
    rows = gold_rows(ROOT / "data" / "processed" / "conll2003" / "test.jsonl", inputs)
    train = gold_rows(ROOT / "data" / "processed" / "conll2003" / "train.jsonl", inputs)
    known = {tuple(word.casefold() for word in row["tokens"][start:end])
             for row in train.values() for start, end, _ in bio_to_spans(row["tags"])}
    paths = paths_for("original", baseline, temporal, robust)
    answers = {method: answer_rows(path, rows, inputs, manifest) for method, path in paths.items()}
    familiar = {method: familiarity(rows, answer, known) for method, answer in answers.items()}
    print(f"Familiarity: {len(familiar)} methods on {len(rows)} complete CoNLL-2003 test sentences", flush=True)

    replacement_rows = {condition: gold_rows(ROOT / "data" / "derived" / "entity_replacement" / "conll2003" /
                                            f"{condition}_v2.jsonl", inputs) for condition in ("original", "replaced")}
    replacement_paths = {condition: paths_for(f"replacement_{condition}", baseline, temporal, robust)
                         for condition in replacement_rows}
    replacement = condition_scores(replacement_rows, replacement_paths, inputs, manifest)
    paired_lora = read_tags_by_id(replacement_paths["original"]["LoRA-LLM"])
    replacement["lora_paired_original_vs_full_changed_ids"] = sorted(
        sample_id for sample_id, tags in paired_lora.items() if tags != answers["LoRA-LLM"][sample_id]
    )

    variants = {"original": rows, **{condition: gold_rows(ROOT / "data" / "derived" / "perturbations" /
                "conll2003" / f"{condition}_v1.jsonl", inputs)
                for condition in ("lowercase", "spelling_noise", "no_punctuation")}}
    variant_paths = {condition: paths if condition == "original" else paths_for(f"{condition}_v1", baseline, temporal, robust)
                     for condition in variants}
    perturbations = condition_scores(variants, variant_paths, inputs, manifest)
    news = {}
    for dataset in ("conll2003", "conllpp"):
        gold = rows if dataset == "conll2003" else gold_rows(ROOT / "data" / "processed" / dataset / "test.jsonl", inputs)
        subset_path = ROOT / "data" / "subsets" / dataset / "test_300.txt"
        ids = read_subset_ids(subset_path)
        if len(ids) != 300 or set(ids) - gold.keys():
            raise ValueError(f"{relative(subset_path)}: expected 300 known test IDs")
        remember(subset_path, inputs, records=len(ids))
        news[dataset] = {"models": {}}
        for method, path in paths_for(f"temporal_{dataset}", baseline, temporal, robust).items():
            selected = {sample_id: gold[sample_id] for sample_id in ids}
            prediction = answer_rows(path, gold, inputs, manifest, subset_ids=ids)
            _, counts = compare({key: row["tags"] for key, row in selected.items()}, prediction)
            news[dataset]["models"][method] = {
                "scores": score_counts(np.array([[counts[key][name] for name in COUNTS] for key in sorted(ids)])),
                "predictions": relative(path),
            }
    print("Replacement, full-test perturbations and both fixed-300 news subsets scored", flush=True)
    ensemble = evaluate_ensembles({key: row["tags"] for key, row in rows.items()}, answers, GROUPS)
    from eval.ensemble_ner import majority_answers
    predicted_spans = {method: {sample_id: bio_to_spans(tags) for sample_id, tags in answer.items()}
                       for method, answer in answers.items()}
    gold = {key: row["tags"] for key, row in rows.items()}
    for group in ensemble["groups"].values():
        tags = majority_answers(gold, predicted_spans, group["members"], group["votes_required"])
        _, counts = compare(gold, tags)
        intervals = score_counts(np.array([[counts[key][name] for name in COUNTS] for key in sorted(rows)]))
        group["scores"]["bootstrap_ci95"] = intervals["bootstrap_ci95"]
    return {
        "source_main_revision": MAIN_REVISION, "source_b_prediction_revision": B_REVISION,
        "source_c_prediction_revision": MAIN_REVISION,
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "libraries": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "sklearn-crfsuite", "python-crfsuite")}},
        "metric": "Exact entity-level micro P/R/F1 on identical sample IDs and four news labels.",
        "entity_types": TYPES, "bootstrap": {**BOOTSTRAP, "unit": "sentence", "interval": "95% percentile"},
        "methods": methods, "inputs": inputs,
        "protocol": {
            "familiarity": "Whole casefolded entity token sequence in any CoNLL-2003 training gold entity, irrespective of its type. False positives use their predicted name.",
            "familiarity_limit": "Seen/unseen uses one common CoNLL-2003 labelled-training reference, not each method's actual exposure, its pretraining corpus or world knowledge. LoRA used only 10% of that training split.",
            "classic_predictions": "Reconstructed locally using the official main code and selected parameters. This report scores the supplied answers directly; small M1 numerical differences from a previous machine's aggregate CSV are resolved by the reconstructed answers.",
            "checksum": "Both actual-byte and LF-normalized SHA-256 values are recorded. B's manifest checks accept exact bytes or CRLF-to-LF normalization only.",
            "neural_seed": "Seed 0 selected in advance for every single-answer comparison; training configurations were selected on validation data by B.",
            "llm": "Qwen2.5-7B, 10 similar retrieved training examples, no guideline; selected using C's fixed validation-300 results.",
            "ensemble": "The four member groups were fixed before scoring; all groups are reported, with strict typed-span majority.",
            "ensemble_reference": "Difference from the group's best observed constituent; bootstrap interval holds that reference fixed and does not account for selecting it.",
            "temporal": "All methods use the same 300 fixed IDs per corpus because C supplies CoNLL++ only at this scope.",
            "coverage": "All ten methods participate in familiarity, paired replacement, full-test perturbations, fixed-300 temporal scoring and error overlap.",
            "replacement_original": "Use the separately delivered paired-original answers for all methods. The new LoRA paired original differs from its earlier full-original output on two sentences; changed IDs are recorded without attributing an unverified cause.",
        },
        "familiarity": {"scored_records": len(rows), "scope": "full", "methods": familiar},
        "replacement": {**replacement, "scope": "paired_500", "version": "v2"},
        "temporal": {"scope": "test_300", "scored_records_per_dataset": 300, "datasets": news},
        "perturbations": {**perturbations, "version": "v1"},
        "ensemble": {"scored_records": len(rows), "scope": "full", **ensemble},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        result = build()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if (args.output.resolve() in {ROOT / path for path in result["inputs"]}
                or args.output.resolve().is_relative_to(ROOT / "preds")
                or args.output.resolve().is_relative_to(ROOT / "data")):
            raise ValueError("--output must be outside prediction and data inputs")
        with TemporaryDirectory(dir=args.output.parent) as temporary:
            staged = Path(temporary) / args.output.name
            staged.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            staged.replace(args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(args.output)


if __name__ == "__main__":
    main()
