"""Benchmark the three in-domain M0–M2 runs through their production CPU paths."""

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter

from threadpoolctl import threadpool_info, threadpool_limits

from eval.score_ner import read_tags_by_id, score_ner
from models.classic import m0_dictionary, m1_logistic, m2_crf, run_baselines


DATASETS = ("conll2003", "wnut17", "mit_restaurant")
RESULTS = run_baselines.RESULTS.with_name("m0_m2_timing.json")
THREADS = 2
WARMUPS = 1
VERSIONS = ("numpy", "scipy", "scikit-learn", "threadpoolctl", "sklearn-crfsuite", "python-crfsuite")


def environment():
    processor = platform.processor() or platform.machine()
    if os.name == "nt":
        import winreg

        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return {
        "device": "CPU",
        "processor": processor,
        "logical_cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "library_versions": {name: version(name) for name in VERSIONS},
        "requested_native_thread_limit": THREADS,
        "native_thread_pools": [
            {key: pool.get(key) for key in ("prefix", "internal_api", "version", "num_threads")}
            for pool in threadpool_info()
        ],
    }


def benchmark_predictions(predict, fit_seconds, path, gold, expected, token_count, repetitions):
    for _ in range(WARMUPS):
        predict(path)
    times = []
    for _ in range(repetitions):
        started = perf_counter()
        predict(path)
        times.append(perf_counter() - started)

    scores = score_ner(gold, read_tags_by_id(path))
    if any(value != expected[key] for key, value in scores.items()):
        raise ValueError(f"{path.name}: full-test scores differ from the official baseline")
    typical = median(times)
    return {
        "fit_seconds": fit_seconds,
        "inference_seconds_median": typical,
        "inference_sentences_per_second": len(gold) / typical,
        "inference_tokens_per_second": token_count / typical,
        "inference_seconds_runs": times,
        "full_test_f1": scores["f1"],
        "gpu_vram_mb": None,
    }


def benchmark_dataset(dataset, reference, directory, repetitions):
    source, shared_pos, _ = run_baselines.GROUPS[dataset]
    train_path, train_by_id = run_baselines.load_rows(source, "train", shared_pos)
    test_path, test_by_id = run_baselines.load_rows(dataset, "test", shared_pos)
    training, test = list(train_by_id.values()), list(test_by_id.values())
    gold = {row["sample_id"]: row["tags"] for row in test}
    test_tokens = sum(len(row["tokens"]) for row in test)
    features = reference["m1_m2_features"]
    selection = run_baselines.m2_selection(dataset, train_path, features)
    if selection["parameters"] != reference["m2"]["parameters"]:
        raise ValueError(f"{dataset}: selected parameters differ from the official baseline")
    expected = reference["targets"][dataset]["full"]["models"]
    print(f"[{dataset}] {len(training):,} training sentences, {len(test):,} test sentences", flush=True)

    def timed_predictions(model, predict, fit_seconds):
        return benchmark_predictions(
            predict, fit_seconds, directory / f"{model}.jsonl", gold, expected[model], test_tokens, repetitions,
        )

    started = perf_counter()
    dictionary, _ = m0_dictionary.build_dictionary(training)
    m0_seconds = perf_counter() - started
    m0 = timed_predictions("m0", lambda path: m0_dictionary.write_predictions(path, test, dictionary), m0_seconds)
    m0.update(parameter_count=len(dictionary), parameter_count_kind="dictionary entries")

    started = perf_counter()
    vectorizer, logistic = m1_logistic.train_model(training, c=reference["m1"]["c"])
    m1_seconds = perf_counter() - started
    m1 = timed_predictions(
        "m1", lambda path: m1_logistic.write_predictions(path, test, vectorizer, logistic), m1_seconds,
    )
    coefficients, intercepts = int(logistic.coef_.size), int(logistic.intercept_.size)
    m1.update(
        parameter_count=coefficients + intercepts, parameter_count_kind="coefficients and intercepts",
        coefficients=coefficients, intercepts=intercepts,
    )

    started = perf_counter()
    crf = m2_crf.train_model(training, directory / "model.crfsuite", **selection["parameters"])
    m2_seconds = perf_counter() - started
    try:
        m2 = timed_predictions("m2", lambda path: m2_crf.write_predictions(path, test, crf), m2_seconds)
        state, transitions = len(crf.state_features_), len(crf.transition_features_)
        m2.update(
            parameter_count=state + transitions, parameter_count_kind="stored CRF state and transition weights",
            state_weights=state, transition_weights=transitions,
        )
    finally:
        crf.tagger_.close()

    models = {"m0": m0, "m1": m1, "m2": m2}
    for model, metrics in models.items():
        print(
            f"[{dataset}/{model}] fit {metrics['fit_seconds']:.3f}s, "
            f"{metrics['inference_sentences_per_second']:,.0f} sentences/s, F1={metrics['full_test_f1']:.4%}",
            flush=True,
        )
    return {
        "training_file": run_baselines.relative(train_path),
        "test_file": run_baselines.relative(test_path),
        "training_records": len(training),
        "training_tokens": sum(len(row["tokens"]) for row in training),
        "test_records": len(test),
        "test_tokens": test_tokens,
        "features": features,
        "m1_parameters": {key: reference["m1"][key] for key in ("c", "tol", "solver", "max_iterations")},
        "m2_parameters": selection["parameters"],
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3, help="Timed full-test inference runs per model")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    reference = json.loads(run_baselines.RESULTS.read_text(encoding="utf-8"))
    if reference["status"] != "complete":
        raise ValueError("the official baseline report must be complete")
    report = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference": run_baselines.relative(run_baselines.RESULTS),
        "protocol": {
            "schedule": "One model operation at a time within this process. Wall times depend on background load.",
            "timer": "time.perf_counter wall time",
            "fit_repetitions": 1,
            "inference_warmups": WARMUPS,
            "inference_repetitions": args.repetitions,
            "fit_scope": "Dictionary construction or feature extraction/vectorization and model fitting, including the CRF model file.",
            "inference_scope": "Full-test production prediction, including feature extraction where applicable, BIO repair and temporary JSONL output.",
            "excluded_from_timing": "Dataset loading, POS generation, cross-validation, scoring and parameter inspection.",
            "gpu_vram_mb": "Null for CPU runs.",
        },
        "datasets": {},
    }
    with TemporaryDirectory(dir=RESULTS.parent) as staging, threadpool_limits(limits=THREADS):
        report["environment"] = environment()
        directory = Path(staging)
        for dataset in DATASETS:
            work = directory / dataset
            work.mkdir()
            report["datasets"][dataset] = benchmark_dataset(dataset, reference["groups"][dataset], work, args.repetitions)
        output = directory / RESULTS.name
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        output.replace(RESULTS)
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
