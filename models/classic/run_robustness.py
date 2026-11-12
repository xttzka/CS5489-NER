"""Compare M0–M2 on fixed paired entity replacements and text perturbations."""

import argparse
import json
from functools import partial
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits

from data.make_entity_replacement import OUTPUT_DIR as PAIR_DIR, VERSION, read_rows
from data.make_perturbation_sets import OUTPUT as PERTURBATIONS, SOURCE as CONLL_TEST, write_jsonl
from eval.baseline_statistics import COUNTS, METRICS, MODELS, bootstrap_counts, compare, metric_values
from eval.score_ner import bio_to_spans, read_tags_by_id
from eval.score_transfer import score_transfer
from models.classic import m0_dictionary, m1_logistic, m2_crf
from models.classic.m2_pos_validation import C1, C2, POS_MODEL, load_pos_rows
from models.classic.run_baselines import load_rows, read_m1_selection, relative


PROJECT = m0_dictionary.PROJECT_DIR
RESULTS = PROJECT / "results" / "m0_m2_robustness.json"
MANIFEST = PERTURBATIONS / "manifest_v1.json"
ENTITY_TYPES = ("PER", "LOC", "ORG", "MISC")
BOOTSTRAP = {"resamples": 1000, "seed": 42}


def load_perturbations():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inputs = {"original": CONLL_TEST}
    for name in ("lowercase", "spelling_noise", "no_punctuation"):
        inputs[name] = PERTURBATIONS / f"{name}_v1.jsonl"

    variants = {}
    original_ids = None
    for condition, path in inputs.items():
        expected = manifest["source_sha256"] if condition == "original" else manifest["sha256"][path.name]
        if sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{path}: file differs from the v1 perturbation manifest")
        rows = load_pos_rows(path, "test")
        ids = [row["sample_id"] for row in rows]
        if original_ids is None:
            original_ids = ids
        if ids != original_ids or len(ids) != manifest["sentences_per_variant"] or len(set(ids)) != len(ids):
            raise ValueError(f"{path}: expected the same unique IDs as the complete test split")
        variants[condition] = (path, rows)
    return variants


def load_replacements(version, test_rows):
    original_path = PAIR_DIR / f"original_{version}.jsonl"
    replaced_path = PAIR_DIR / f"replaced_{version}.jsonl"
    changes_path = PAIR_DIR / f"replacements_{version}.jsonl"
    original = load_pos_rows(original_path, "test")
    replaced = load_pos_rows(replaced_path, "test")
    changes = list(read_rows(changes_path))
    test_by_id = {row["sample_id"]: row for row in test_rows}
    ids = [change["sample_id"] for change in changes]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"{changes_path}: expected one replacement per unique original sentence")

    for before, after, change in zip(original, replaced, changes, strict=True):
        sample_id = change["sample_id"]
        if before != test_by_id.get(sample_id) or any(
            before[key] != after[key] for key in before if key != "tokens"
        ):
            raise ValueError(f"{sample_id}: paired metadata or labels differ from the complete test split")
        start, end, kind = change["start"], change["end"], change["entity_type"]
        tokens, modified = before["tokens"], after["tokens"]
        if (
            len(tokens) != len(modified)
            or (start, end, kind) not in bio_to_spans(before["tags"])
            or tokens[start:end] != change["original_tokens"]
            or modified[start:end] != change["replacement_tokens"]
            or tokens[:start] != modified[:start]
            or tokens[end:] != modified[end:]
        ):
            raise ValueError(f"{sample_id}: paired tokens do not match the replacement log")
    return {"original": (original_path, original), "replaced": (replaced_path, replaced)}, changes_path, changes


def predict(model, writer, rows, dataset, condition, stage, pending):
    staged = stage / f"{model}_{condition}.jsonl"
    output = PROJECT / "preds" / dataset / f"{model}_robustness_{condition}" / "0.jsonl"
    repairs = writer(staged, rows)
    pending[staged] = output
    details = {"predictions": relative(output)}
    if model != "m0":
        details.update(raw_bio_error_tokens=repairs[0], raw_bio_error_sentences=repairs[1])
    return read_tags_by_id(staged), details


def evaluate_experiment(variants, writers, experiment, stage, pending):
    conditions = {name: {"input": relative(path), "models": {}} for name, (path, _) in variants.items()}
    gold = {name: {row["sample_id"]: row["tags"] for row in rows} for name, (_, rows) in variants.items()}
    ids = sorted(gold["original"])
    answers = {}
    for model, writer in writers.items():
        answers[model] = {}
        counts = []
        for name, (_, rows) in variants.items():
            prediction, details = predict(model, writer, rows, "conll2003", f"{experiment}_{name}", stage, pending)
            answers[model][name] = prediction
            scores, sentence_counts = compare(gold[name], prediction)
            conditions[name]["models"][model] = {"scores": scores, **details}
            counts.append([[sentence_counts[sample_id][key] for key in COUNTS] for sample_id in ids])

        # Every replicate uses the same sentence indices for all versions of a test sentence.
        samples = metric_values(bootstrap_counts(np.asarray(counts).transpose(1, 0, 2), **BOOTSTRAP))
        original = conditions["original"]["models"][model]["scores"]
        for index, (name, condition) in enumerate(conditions.items()):
            row = condition["models"][model]
            scores = row["scores"]
            if scores["tp"] + scores["fn"] != original["tp"] + original["fn"]:
                raise ValueError(f"{experiment}/{name}: expected the same number of gold entities as the original")
            values = samples[:, index]
            intervals = np.quantile(values, (0.025, 0.975), axis=0).T.tolist()
            differences = np.quantile((values - samples[:, 0]) * 100, (0.025, 0.975), axis=0).T.tolist()
            scores["bootstrap_ci95"] = dict(zip(METRICS, intervals, strict=True))
            row["delta_vs_original_pp"] = {key: (scores[key] - original[key]) * 100 for key in METRICS}
            row["delta_vs_original_ci95_pp"] = dict(zip(METRICS, differences, strict=True))
            print(f"{experiment}/{name} {model.upper()} F1={scores['f1']:.4%}", flush=True)
    return conditions, answers


def summarize_targets(items):
    original = sum(item["original_correct"] for item in items)
    replaced = sum(item["replaced_correct"] for item in items)
    both = sum(item["original_correct"] and item["replaced_correct"] for item in items)
    return {
        "targets": len(items), "original_correct": original, "replaced_correct": replaced,
        "both_correct": both, "original_only": original - both, "replaced_only": replaced - both,
        "neither_correct": len(items) - original - replaced + both,
    }


def score_replaced_targets(changes, answers, experiment, stage, pending):
    models = {}
    for model, predictions in answers.items():
        items = []
        for change in changes:
            item = {key: change[key] for key in ("sample_id", "start", "end", "entity_type")}
            span = item["start"], item["end"], item["entity_type"]
            for condition, answer in predictions.items():
                item[f"{condition}_correct"] = span in bio_to_spans(answer[item["sample_id"]])
            items.append(item)
        output = PROJECT / "preds" / "conll2003" / f"{model}_robustness_{experiment}" / "targets.jsonl"
        staged = stage / f"{model}_{experiment}_targets.jsonl"
        write_jsonl(staged, items)
        pending[staged] = output
        models[model] = {
            **summarize_targets(items),
            "per_type": {kind: summarize_targets([item for item in items if item["entity_type"] == kind])
                         for kind in ENTITY_TYPES},
            "target_details": relative(output),
        }
    return {"models": models}


def lowercase_vs_btc(writers, variants, answers, stage, pending):
    btc_path, btc_rows = load_rows("btc", "test", shared_pos=True)
    gold = {
        condition: {row["sample_id"]: row["tags"] for row in variants[condition][1]}
        for condition in ("original", "lowercase")
    }
    btc_gold = {row["sample_id"]: row["tags"] for row in btc_rows.values()}
    models = {}
    for model, writer in writers.items():
        btc_answer, details = predict(model, writer, list(btc_rows.values()), "btc", "transfer", stage, pending)
        scores = {
            name: score_transfer(reference, answers[model][name], target_dataset="conll2003", prediction_schema="conll2003")
            for name, reference in gold.items()
        }
        scores["btc"] = score_transfer(btc_gold, btc_answer, target_dataset="btc", prediction_schema="conll2003")
        models[model] = {
            **scores, **details,
            "lowercase_loss_pp": 100 * (scores["original"]["f1"] - scores["lowercase"]["f1"]),
            "btc_loss_pp": 100 * (scores["original"]["f1"] - scores["btc"]["f1"]),
        }
    return {
        "btc_input": relative(btc_path), "entity_types": ("PER", "LOC", "ORG"), "models": models,
        "interpretation": "Descriptive comparison of full lowercasing with transfer to a different corpus.",
    }


def run(version):
    perturbations = load_perturbations()
    replacements, log, changes = load_replacements(version, perturbations["original"][1])
    train_path, train_rows = load_rows("conll2003", "train", shared_pos=True)
    training = list(train_rows.values())
    m1_config = read_m1_selection("conll2003", train_path)
    m2_params = {"c1": C1, "c2": C2}
    dictionary, ambiguous = m0_dictionary.build_dictionary(training)
    vectorizer, logistic = m1_logistic.train_model(training, c=m1_config["c"])

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=RESULTS.parent) as temp, threadpool_limits(limits=2):
        stage = Path(temp)
        pending = {}
        crf = m2_crf.train_model(training, stage / "model.crfsuite", **m2_params)
        writers = dict(zip(MODELS, (
            partial(m0_dictionary.write_predictions, dictionary=dictionary),
            partial(m1_logistic.write_predictions, vectorizer=vectorizer, classifier=logistic),
            partial(m2_crf.write_predictions, classifier=crf),
        ), strict=True))
        replacement_name = f"replacement_{version}"
        try:
            replaced_conditions, replaced_answers = evaluate_experiment(
                replacements, writers, replacement_name, stage, pending,
            )
            perturbed_conditions, perturbed_answers = evaluate_experiment(
                perturbations, writers, "perturbation_v1", stage, pending,
            )
            btc = lowercase_vs_btc(writers, perturbations, perturbed_answers, stage, pending)
        finally:
            crf.tagger_.close()

        report = {
            "training_input": relative(train_path), "training_records": len(training),
            "features": {"pos_model": POS_MODEL, "use_chunk_tags": False, "context_window": 1,
                         "pos_frozen_from_original": True},
            "entity_types": ENTITY_TYPES,
            "bootstrap": {**BOOTSTRAP, "unit": "sentence", "interval": "95% percentile"},
            "models": {
                "m0": {"dictionary_entries": len(dictionary), "ambiguous_names_excluded": ambiguous},
                "m1": {"c": logistic.C, "tol": logistic.tol, "max_iterations": logistic.max_iter,
                       "iterations": int(logistic.n_iter_[0])},
                "m2": {"parameters": m2_params},
            },
            "replacement": {
                "version": version, "replacement_log": relative(log), "paired_test_sentences": len(changes),
                "conditions": replaced_conditions,
                "replacement_targets": score_replaced_targets(changes, replaced_answers, replacement_name, stage, pending),
            },
            "perturbations": {
                "version": "v1", "manifest": relative(MANIFEST),
                "test_records_per_condition": len(perturbations["original"][1]),
                "conditions": perturbed_conditions,
            },
            "lowercase_vs_btc": btc,
        }
        staged_report = stage / RESULTS.name
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for staged, output in pending.items():
            output.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(output)
        staged_report.replace(RESULTS)
    print(f"Results: {RESULTS}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replacement-version", choices=(VERSION,), default=VERSION)
    args = parser.parse_args()
    run(args.replacement_version)


if __name__ == "__main__":
    main()
