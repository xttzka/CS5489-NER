"""Compare POS-only M2 with external gazetteer and GloVe cluster features."""

import io
import json
from functools import partial
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from data.make_entity_index import name_key, read_entities
from eval.score_m2_familiarity import RESULTS as FAMILIARITY_REPORT
from eval.score_m2_familiarity import TRANSFER_REPORT, load_spans, score_groups
from models.classic.features import sentence_features
from models.classic.m2_feature_ablation import PARAMETERS, fit_feature_model, predict_feature_rows
from models.classic.m2_pos_validation import (
    C1, C2, POS_MODEL, POS_ROOT, PROJECT_DIR, load_pos_rows, verify_alignment,
)


RESULTS = PROJECT_DIR / "results" / "m2_extra_features.json"
EXTERNAL = PROJECT_DIR / "data" / "external"
DATASETS = ("conll2003", "conllpp")
LABELS = {kind: kind for kind in ("PER", "LOC", "ORG", "MISC")}
K_VALUES = (100, 500, 1000)
SEED = 42
THREADS = 2
CITIES = EXTERNAL / "gazetteers" / "cities15000.zip"
COUNTRIES = EXTERNAL / "gazetteers" / "countryInfo.txt"
GLOVE = EXTERNAL / "glove" / "glove.6B.100d.txt"
RESOURCES = {
    CITIES: (
        "https://download.geonames.org/export/dump/cities15000.zip",
        "0df068fe1eb445f18655559b9d80fdd45ac88535d8c27b7b4f5ce5e66b1107b3",
        "CC BY 4.0",
    ),
    COUNTRIES: (
        "https://download.geonames.org/export/dump/countryInfo.txt",
        "79f8f59ea01cbb40438bf1856dd6c2ff201f03f8fc72d0530b2646aaf603ef04",
        "CC BY 4.0",
    ),
    GLOVE: (
        "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
        "be4367dd257eb945217234f16307c5c74236b648a222cc0b4ffd0dda6a3350b6",
        "ODC PDDL 1.0",
    ),
}


def verified_resource(path):
    source, expected, license_name = RESOURCES[path]
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"{path}: SHA-256 differs from the experiment resource")
    return {"url": source, "sha256": expected, "license": license_name}


def gazetteer_names():
    names = set()
    with ZipFile(CITIES) as archive, io.TextIOWrapper(archive.open("cities15000.txt"), encoding="utf-8") as rows:
        for row in rows:
            fields = row.rstrip("\r\n").split("\t")
            if len(fields) != 19:
                raise ValueError("unexpected GeoNames cities schema")
            for name in fields[1:3]:
                if name.strip():
                    names.add(tuple(name.casefold().split()))
    with COUNTRIES.open(encoding="utf-8") as rows:
        for row in rows:
            if row.strip() and not row.startswith("#"):
                name = row.split("\t")[4]
                if name.strip():
                    names.add(tuple(name.casefold().split()))
    return names


def with_gazetteer(row, names, max_tokens):
    features = sentence_features(row)
    tokens = tuple(token.casefold() for token in row["tokens"])
    start = 0
    while start < len(tokens):
        stop = min(len(tokens), start + max_tokens)
        while stop > start and tokens[start:stop] not in names:
            stop -= 1
        if stop > start:
            features[start]["gazetteer:position"] = "start"
            for index in range(start + 1, stop):
                features[index]["gazetteer:position"] = "inside"
        start = max(start + 1, stop)
    return features


def vocabulary(rows):
    return {token.lower() for row in rows for token in row["tokens"]}


def load_vectors(wanted):
    vectors = {}
    with GLOVE.open(encoding="utf-8") as source:
        for row in source:
            word, values = row.split(maxsplit=1)
            if word in wanted:
                vector = np.fromstring(values, dtype=np.float32, sep=" ")
                if vector.shape != (100,) or not np.isfinite(vector).all():
                    raise ValueError(f"invalid GloVe vector for {word!r}")
                vectors[word] = vector
    return vectors


def with_cluster(row, cluster_ids):
    features = sentence_features(row)
    for token, token_features in zip(row["tokens"], features, strict=True):
        cluster_id = cluster_ids.get(token.lower())
        if cluster_id is not None:
            token_features["glove:cluster"] = cluster_id
    return features


def baseline_scores(training_path, train_count, seen_names):
    transfer = json.loads(TRANSFER_REPORT.read_text(encoding="utf-8"))
    familiarity = json.loads(FAMILIARITY_REPORT.read_text(encoding="utf-8"))
    if (
        PARAMETERS != {"c1": C1, "c2": C2}
        or transfer["parameters"] != PARAMETERS
        or transfer["features"] != {"pos_model": POS_MODEL, "use_chunk_tags": False}
        or transfer["source_dataset"] != DATASETS[0]
        or transfer["training_input"] != training_path.relative_to(PROJECT_DIR).as_posix()
        or transfer["training_records"] != train_count
        or familiarity["model_report"] != TRANSFER_REPORT.relative_to(PROJECT_DIR).as_posix()
        or set(familiarity["news_four_types"]["entity_types"]) != LABELS.keys()
    ):
        raise ValueError("saved POS-only baseline does not match the feature experiment")

    baseline = {}
    for dataset in DATASETS:
        target = transfer["targets"][dataset]
        gold, predicted, count = load_spans(dataset, target, seen_names)
        scores = score_groups(gold, predicted, LABELS, LABELS)
        cached = familiarity["news_four_types"]["datasets"][dataset]
        if cached["predictions"] != target["predictions"] or any(cached[k] != v for k, v in scores.items()):
            raise ValueError(f"{dataset}: baseline metrics differ from the saved four-type scores")
        baseline[dataset] = {"records": count, "predictions": target["predictions"], **scores}
    return baseline


def evaluate(name, extract, training, targets, seen_names, baseline):
    print(f"Training {name}", flush=True)
    scored = {}
    pending = {}
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        directory = Path(staging)
        classifier = fit_feature_model(training, directory / "model.crfsuite", extract)
        try:
            for dataset, rows in targets.items():
                predictions, bio_counts = predict_feature_rows(rows, classifier, extract)
                staged = directory / f"{dataset}.jsonl"
                with staged.open("w", encoding="utf-8", newline="\n") as output:
                    for sample_id, tags in predictions.items():
                        output.write(json.dumps({"sample_id": sample_id, "tags": tags}) + "\n")
                gold_path = PROJECT_DIR / "data" / "processed" / dataset / "test.jsonl"
                gold, predicted, count = load_spans(dataset, {"gold": gold_path, "predictions": staged}, seen_names)
                scores = score_groups(gold, predicted, LABELS, LABELS)
                path = PROJECT_DIR / "preds" / dataset / f"m2_extra_{name}" / "0.jsonl"
                pending[staged] = path
                scored[dataset] = {
                    "records": count,
                    "predictions": path.relative_to(PROJECT_DIR).as_posix(),
                    **scores,
                    **bio_counts,
                    "f1_delta_vs_baseline_pp": {
                        group: (scores[group]["f1"] - baseline[dataset][group]["f1"]) * 100
                        for group in ("overall", "seen", "unseen")
                    },
                }
                print(f"{name} | {dataset}: F1 {scores['overall']['f1']:.2%}, unseen {scores['unseen']['f1']:.2%}", flush=True)
        finally:
            classifier.tagger_.close()
        for staged, path in pending.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            staged.replace(path)
    return scored


def save_report(report):
    with TemporaryDirectory(dir=RESULTS.parent) as staging:
        path = Path(staging) / RESULTS.name
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        path.replace(RESULTS)


def main():
    training_path = POS_ROOT / DATASETS[0] / "train.jsonl"
    training = load_pos_rows(training_path, "train", dataset=DATASETS[0])
    verify_alignment(PROJECT_DIR / "data" / "processed" / DATASETS[0] / "train.jsonl", training, "train")
    targets = {}
    for dataset in DATASETS:
        targets[dataset] = load_pos_rows(POS_ROOT / dataset / "test.jsonl", "test", dataset=dataset)
        verify_alignment(PROJECT_DIR / "data" / "processed" / dataset / "test.jsonl", targets[dataset], "test")
    seen_names = {name_key(tokens) for _, _, _, _, tokens in read_entities(DATASETS[0], "train")}
    baseline = baseline_scores(training_path, len(training), seen_names)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "in_progress",
        "parameters": PARAMETERS,
        "features": {"pos_model": POS_MODEL, "use_chunk_tags": False},
        "training": training_path.relative_to(PROJECT_DIR).as_posix(),
        "entity_types": list(LABELS),
        "baseline_report": FAMILIARITY_REPORT.relative_to(PROJECT_DIR).as_posix(),
        "baseline": baseline,
        "resources": {
            path.relative_to(PROJECT_DIR).as_posix(): verified_resource(path)
            for path in (CITIES, COUNTRIES)
        },
        "gazetteer_snapshot_retrieved": "2026-11-10",
        "gazetteer_matching": "Casefolded city, ASCII city and country names; longest match at each token start.",
        "cluster_config": {
            "algorithm": "KMeans (Lloyd)", "random_state": SEED, "n_init": 1, "max_iter": 100,
            "k": list(K_VALUES), "fit_vocabulary": "CoNLL-2003 training tokens only", "dimensions": 100,
        },
        "conditions": {},
    }
    names = gazetteer_names()
    report["gazetteer_names"] = len(names)
    gazetteer_extract = partial(with_gazetteer, names=names, max_tokens=max(map(len, names)))
    with threadpool_limits(limits=THREADS):
        report["conditions"]["gazetteer"] = evaluate("gazetteer", gazetteer_extract, training, targets, seen_names, baseline)
        save_report(report)

        report["resources"][GLOVE.relative_to(PROJECT_DIR).as_posix()] = verified_resource(GLOVE)
        train_words = vocabulary(training)
        target_words = {dataset: vocabulary(rows) for dataset, rows in targets.items()}
        vectors = load_vectors(train_words.union(*target_words.values()))
        fit_words = sorted(train_words & vectors.keys())
        if len(fit_words) < max(K_VALUES):
            raise ValueError("too few GloVe-covered training words for the fixed cluster counts")
        fit_matrix = np.stack([vectors[word] for word in fit_words])
        lookup_words = sorted(vectors)
        lookup_matrix = np.stack([vectors[word] for word in lookup_words])
        report["glove_type_coverage"] = {
            name: {"covered": len(words & vectors.keys()), "total": len(words)}
            for name, words in {"train": train_words, **target_words}.items()
        }
        for k in K_VALUES:
            print(f"Fitting k={k} to {len(fit_words):,} GloVe-covered training word types", flush=True)
            cluster_model = KMeans(n_clusters=k, random_state=SEED, n_init=1, max_iter=100)
            cluster_model.fit(fit_matrix)
            cluster_ids = {
                word: str(cluster_id)
                for word, cluster_id in zip(lookup_words, cluster_model.predict(lookup_matrix), strict=True)
            }
            name = f"glove_{k}"
            extract = partial(with_cluster, cluster_ids=cluster_ids)
            report["conditions"][name] = evaluate(name, extract, training, targets, seen_names, baseline)
            save_report(report)

    report["status"] = "complete"
    save_report(report)
    print(f"Results: {RESULTS}")


if __name__ == "__main__":
    main()
