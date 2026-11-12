"""Cluster CoNLL-2003 gold entity spans with GloVe and optional BERT vectors.

BERT JSONL rows require sample_id, start, end and a numeric vector. Offsets
index the original tokens from zero, with end excluded. An optional
entity_type must match the gold span. Supply both BERT files to compare the
three representations on their intersection. GloVe alone runs the pilot.
"""

import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from analysis.data_exploration import ROOT
from eval.score_ner import bio_to_spans


DEFAULT_GOLD = ROOT / "data" / "processed" / "conll2003" / "test.jsonl"
DEFAULT_GLOVE = ROOT / "data" / "external" / "glove" / "glove.6B.100d.txt"
GLOVE_HASH = "be4367dd257eb945217234f16307c5c74236b648a222cc0b4ffd0dda6a3350b6"
TYPES = {"PER", "LOC", "ORG", "MISC"}
SEED = 42
N_INIT = 10


def reference(path):
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)


def jsonl(path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                yield row


def read_gold(path):
    spans, ids = {}, set()
    for row in jsonl(path):
        sample_id, words, tags = (row[key] for key in ("sample_id", "tokens", "tags"))
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in ids:
            raise ValueError(f"{path}: invalid or repeated sample ID {sample_id!r}")
        if len(words) != len(tags):
            raise ValueError(f"{sample_id}: token and tag counts differ")
        ids.add(sample_id)
        for start, end, kind in bio_to_spans(tags):
            if kind not in TYPES:
                raise ValueError(f"{sample_id}: unexpected entity type {kind!r}")
            spans[sample_id, start, end] = (kind, words[start:end])
    if not spans:
        raise ValueError(f"{path}: no gold entities")
    return spans


def glove_entity_vectors(path, gold):
    wanted = {word.lower().encode("utf-8") for _, words in gold.values() for word in words}
    word_vectors = {}
    digest = sha256()
    dimensions = None
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            word, _, values = line.partition(b" ")
            if word not in wanted:
                continue
            vector = np.fromstring(values, dtype=np.float32, sep=" ")
            if dimensions is None:
                dimensions = len(vector)
            if not dimensions or len(vector) != dimensions or not np.isfinite(vector).all() or word in word_vectors:
                raise ValueError(f"{path}: invalid or repeated word vector for {word!r}")
            word_vectors[word] = vector
    checksum = digest.hexdigest()
    if path.resolve() == DEFAULT_GLOVE.resolve() and (checksum != GLOVE_HASH or dimensions != 100):
        raise ValueError(f"{path}: expected the verified 100-dimensional GloVe 6B file")

    vectors = {}
    partial = 0
    for key, (_, words) in gold.items():
        known = [word_vectors[word.lower().encode("utf-8")] for word in words
                 if word.lower().encode("utf-8") in word_vectors]
        if known:
            vectors[key] = np.mean(known, axis=0)
            partial += len(known) < len(words)
    return vectors, {
        "input": reference(path), "sha256": checksum, "dimensions": dimensions,
        "partially_out_of_vocabulary_spans": partial,
        "fully_out_of_vocabulary_spans": len(gold) - len(vectors),
        "pooling": "Mean of the available lowercase token vectors; exclude spans with no covered tokens.",
    }


def read_bert_vectors(path, gold):
    vectors = {}
    dimensions = None
    for row in jsonl(path):
        key = row["sample_id"], row["start"], row["end"]
        if not isinstance(key[0], str) or any(type(offset) is not int for offset in key[1:]):
            raise ValueError(f"{path}: invalid span ID {key!r}")
        if key not in gold or key in vectors:
            raise ValueError(f"{path}: unknown or repeated gold span {key!r}")
        if row.get("entity_type", gold[key][0]) != gold[key][0]:
            raise ValueError(f"{path}: gold type differs for {key!r}")
        values = row["vector"]
        if not isinstance(values, list) or not values or any(type(value) not in (int, float) for value in values):
            raise ValueError(f"{path}: expected a numeric vector for {key!r}")
        vector = np.asarray(values, dtype=np.float32)
        if dimensions is None:
            dimensions = len(vector)
        if len(vector) != dimensions or not np.isfinite(vector).all():
            raise ValueError(f"{path}: inconsistent dimension or nonfinite vector for {key!r}")
        vectors[key] = vector
    return vectors, {"input": reference(path), "sha256": sha256(path.read_bytes()).hexdigest(), "dimensions": dimensions}


def cluster(vectors, keys, gold, misc_k):
    matrix = np.stack([vectors[key] for key in keys])
    misc_indices = [i for i, key in enumerate(keys) if gold[key][0] == "MISC"]
    misc_matrix = matrix[misc_indices]
    if len(np.unique(matrix, axis=0)) < len(TYPES) or len(np.unique(misc_matrix, axis=0)) < misc_k:
        raise ValueError(f"need at least four distinct full vectors and {misc_k} distinct MISC vectors")
    labels = KMeans(n_clusters=len(TYPES), n_init=N_INIT, random_state=SEED).fit_predict(matrix)
    misc_labels = KMeans(n_clusters=misc_k, n_init=N_INIT, random_state=SEED).fit_predict(misc_matrix)
    gold_types = [gold[key][0] for key in keys]
    counts = {str(i): dict(Counter(kind for kind, label in zip(gold_types, labels, strict=True) if label == i))
              for i in range(len(TYPES))}
    metrics = {
        "ari": float(adjusted_rand_score(gold_types, labels)),
        "nmi": float(normalized_mutual_info_score(gold_types, labels)),
        "cluster_gold_type_counts": counts,
        "misc_cluster_sizes": {str(i): int(np.count_nonzero(misc_labels == i)) for i in range(misc_k)},
    }
    return metrics, dict(zip(keys, map(int, labels), strict=True)), {
        keys[i]: int(label) for i, label in zip(misc_indices, misc_labels, strict=True)
    }


def save(report_path, assignment_path, report, assignments):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=report_path.parent) as result_temp, TemporaryDirectory(dir=assignment_path.parent) as review_temp:
        result_stage = Path(result_temp) / report_path.name
        review_stage = Path(review_temp) / assignment_path.name
        result_stage.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with review_stage.open("w", encoding="utf-8", newline="\n") as target:
            for row in assignments:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        review_stage.replace(assignment_path)
        result_stage.replace(report_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--glove", type=Path, default=DEFAULT_GLOVE)
    parser.add_argument("--base-bert", type=Path)
    parser.add_argument("--fine-tuned-bert", type=Path)
    parser.add_argument("--misc-k", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assignments", type=Path)
    args = parser.parse_args()
    if (args.base_bert is None) != (args.fine_tuned_bert is None):
        parser.error("supply --base-bert and --fine-tuned-bert together")
    if args.misc_k < 2:
        parser.error("--misc-k must be at least 2")

    mode = "matched" if args.base_bert else "glove"
    report_path = args.output or ROOT / "results" / f"entity_clustering_{mode}.json"
    assignment_path = args.assignments or ROOT / "output" / "entity_clustering" / f"{mode}_assignments.jsonl"
    inputs = {path.resolve() for path in (args.gold, args.glove, args.base_bert, args.fine_tuned_bert) if path is not None}
    outputs = {report_path.resolve(), assignment_path.resolve()}
    if len(outputs) != 2 or outputs & inputs:
        parser.error("use distinct output paths outside the input files")

    gold = read_gold(args.gold)
    representations = {"glove": glove_entity_vectors(args.glove, gold)}
    if args.base_bert:
        representations["base_bert"] = read_bert_vectors(args.base_bert, gold)
        representations["fine_tuned_bert"] = read_bert_vectors(args.fine_tuned_bert, gold)
    keys = sorted(set.intersection(*(set(vectors) for vectors, _ in representations.values())))
    type_counts = Counter(gold[key][0] for key in keys)
    if type_counts.keys() != TYPES or type_counts["MISC"] < args.misc_k:
        raise ValueError(f"shared spans must cover all four gold types and at least {args.misc_k} MISC entities")

    metrics, all_labels, misc_labels = {}, {}, {}
    with threadpool_limits(limits=2):
        for name, (vectors, metadata) in representations.items():
            scores, all_labels[name], misc_labels[name] = cluster(vectors, keys, gold, args.misc_k)
            metrics[name] = {**metadata, "available_spans": len(vectors), "used_spans": len(keys), **scores}
            print(f"{name}: {len(keys):,} spans, ARI={scores['ari']:.4f}, NMI={scores['nmi']:.4f}")

    assignments = []
    for sample_id, start, end in keys:
        key = sample_id, start, end
        kind, words = gold[key]
        assignments.append({
            "span_id": f"{sample_id}:{start}:{end}", "sample_id": sample_id, "start": start, "end": end,
            "entity_type": kind, "entity": " ".join(words),
            "clusters": {name: labels[key] for name, labels in all_labels.items()},
            "misc_clusters": {name: labels[key] for name, labels in misc_labels.items()} if kind == "MISC" else {},
        })
    report = {
        "mode": mode, "gold": reference(args.gold), "gold_sha256": sha256(args.gold.read_bytes()).hexdigest(),
        "gold_spans": len(gold), "shared_spans": len(keys), "excluded_spans": len(gold) - len(keys),
        "gold_type_counts": dict(sorted(Counter(kind for kind, _ in gold.values()).items())),
        "shared_type_counts": dict(sorted(type_counts.items())),
        "matching": "Identical intersection of (sample_id, start, end) gold spans for every representation.",
        "cluster_config": {"algorithm": "K-means (Lloyd)", "k": len(TYPES), "misc_k": args.misc_k,
                           "n_init": N_INIT, "random_state": SEED, "normalization": "none"},
        "misc_review": "MISC subclusters use cluster sizes and span assignments for manual interpretation.",
        "representations": metrics, "assignments": reference(assignment_path),
    }
    save(report_path, assignment_path, report, assignments)
    print(report_path)
    print(assignment_path)


if __name__ == "__main__":
    main()
