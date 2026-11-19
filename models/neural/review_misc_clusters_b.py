"""Prepare a reproducible, context-rich manual review of MISC subclusters.

Run analysis.cluster_entities first on matched GloVe/base/fine-tuned BERT
vectors. This script checks that file against its gold data, then samples
occurrences within each cluster for human review. Counts are descriptive;
neither the sampling nor K-means assigns a semantic label to a cluster.
Final/test reviews require a valid, pre-existing frozen B evaluation manifest.
"""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

from analysis.cluster_entities import jsonl, read_gold


REPRESENTATIONS = ("glove", "base_bert", "fine_tuned_bert")


def digest(path):
    value = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def validate_manifest(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    declared = obj.pop("manifest_sha256", None)
    actual = sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False).encode()).hexdigest()
    if obj.get("version") != 1 or obj.get("state") != "frozen" or declared != actual:
        raise ValueError("final review requires an unmodified, frozen version-1 B manifest")
    if obj.get("partial") or obj.get("missing_requirements"):
        raise ValueError("a partial manifest cannot authorize held-out review")
    return declared


def safe(value):
    """Keep a single occurrence in one Markdown table cell."""
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True,
                        help="Matched-clustering metadata from analysis.cluster_entities")
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--phase", choices=("validation", "final"), required=True)
    parser.add_argument("--frozen", type=Path, help="Required for a final/test review")
    parser.add_argument("--output", type=Path, required=True, help="Machine-readable JSON")
    parser.add_argument("--markdown", type=Path, required=True, help="Review worksheet")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        if args.sample_size < 1:
            raise ValueError("sample-size must be positive")
        if args.phase == "final" and not args.frozen:
            raise ValueError("--frozen is required before final/test review")
        manifest_hash = validate_manifest(args.frozen) if args.phase == "final" else None
        paths = [args.gold, args.metrics, args.assignments, args.output, args.markdown]
        if args.frozen:
            paths.append(args.frozen)
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError("all inputs and outputs must be distinct")
        if args.output.exists() or args.markdown.exists():
            raise FileExistsError("output files must not already exist")

        metadata = json.loads(args.metrics.read_text(encoding="utf-8"))
        if metadata.get("mode") != "matched" or set(metadata.get("representations", {})) != set(REPRESENTATIONS):
            raise ValueError("metrics must contain all three matched representations")
        if digest(args.gold) != metadata.get("gold_sha256"):
            raise ValueError("gold SHA-256 does not match the clustering metadata")
        k = metadata.get("cluster_config", {}).get("misc_k")
        if type(k) is not int or k < 2:
            raise ValueError("invalid MISC cluster count in metadata")
        contexts = {}
        for row in jsonl(args.gold):
            if args.phase == "validation" and row.get("split") == "test":
                raise ValueError("validation phase cannot read test rows")
            contexts[row["sample_id"]] = row["tokens"]
        gold = read_gold(args.gold)
        groups = {rep: defaultdict(list) for rep in REPRESENTATIONS}
        seen = set()
        n_assignments = 0
        for row in jsonl(args.assignments):
            key = row["sample_id"], row["start"], row["end"]
            if key in seen or key not in gold:
                raise ValueError(f"duplicate or unknown assignment span: {key}")
            seen.add(key)
            n_assignments += 1
            kind, words = gold[key]
            if row.get("entity_type") != kind or row.get("entity") != " ".join(words):
                raise ValueError(f"assignment differs from gold: {key}")
            if kind != "MISC":
                continue
            labels = row.get("misc_clusters")
            if not isinstance(labels, dict) or set(labels) != set(REPRESENTATIONS):
                raise ValueError(f"missing matched MISC cluster labels for {key}")
            sentence = contexts[key[0]]
            occurrence = {"sample_id": key[0], "start": key[1], "end": key[2],
                          "entity": " ".join(words),
                          "context": " ".join(sentence[:key[1]] +
                                              ["[" + " ".join(words) + "]"] +
                                              sentence[key[2]:])}
            for rep in REPRESENTATIONS:
                if type(labels[rep]) is not int or not 0 <= labels[rep] < k:
                    raise ValueError(f"invalid cluster label for {key}: {rep}")
                groups[rep][labels[rep]].append(occurrence)
        if n_assignments != metadata.get("shared_spans"):
            raise ValueError("assignment count does not match clustering metadata")

        result = {"phase": args.phase, "frozen_manifest_sha256": manifest_hash,
                  "gold": str(args.gold), "gold_sha256": digest(args.gold),
                  "metrics": str(args.metrics), "metrics_sha256": digest(args.metrics),
                  "assignments": str(args.assignments), "assignments_sha256": digest(args.assignments),
                  "seed": args.seed, "sample_size_per_cluster": args.sample_size,
                  "sampling": "uniform without replacement over occurrences within each cluster; deterministic seed",
                  "warning": "Cluster IDs do not imply semantic labels. Samples are exploratory and do not establish a causal explanation of model errors.",
                  "representations": {}}
        lines = ["# MISC subcluster review worksheet", "",
                 f"Phase: {args.phase}; seed: {args.seed}; requested sample: {args.sample_size} occurrences per cluster.", "",
                 "Each representation uses the same matched entity spans. Within each cluster the displayed occurrences are sampled uniformly, not chosen to illustrate a particular conclusion. Frequent surface forms and samples have different selection rules.", "",
                 "Read the full examples before assigning a tentative human interpretation. Cluster IDs are arbitrary; comparable IDs across representations need not have comparable meanings. These clusters alone cannot establish why a system makes errors or whether MISC is the lowest-scoring type.", ""]
        for rep in REPRESENTATIONS:
            if set(groups[rep]) != set(range(k)):
                raise ValueError(f"empty or missing {rep} MISC clusters")
            expected = metadata["representations"][rep].get("misc_cluster_sizes", {})
            result["representations"][rep] = {}
            lines.extend([f"## {rep}", ""])
            for cluster in range(k):
                rows = sorted(groups[rep][cluster], key=lambda row: (row["sample_id"], row["start"], row["end"]))
                if int(expected.get(str(cluster), -1)) != len(rows):
                    raise ValueError(f"{rep} cluster {cluster} size differs from metrics")
                # String seeds are stable across Python processes (unlike hash()).
                rng = random.Random(f"{args.seed}:{rep}:{cluster}")
                sample = rng.sample(rows, min(args.sample_size, len(rows)))
                counts = Counter(row["entity"] for row in rows)
                most_common = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]
                result["representations"][rep][str(cluster)] = {
                    "occurrences": len(rows), "distinct_surfaces": len(counts),
                    "frequent_surfaces": [{"entity": name, "count": count} for name, count in most_common],
                    "review_occurrences": sample}
                frequencies = "; ".join(f"{safe(name)} ({count})" for name, count in most_common)
                lines.extend([f"### Cluster {cluster}: {len(rows)} occurrences; {len(counts)} unique surface forms", "",
                              f"Most frequent: {frequencies}", "",
                              "| ID / token span | Entity | Tokenized sentence, entity in brackets |",
                              "| --- | --- | --- |"])
                for row in sample:
                    identifier = f"{row['sample_id']}:{row['start']}:{row['end']}"
                    lines.append(f"| {safe(identifier)} | {safe(row['entity'])} | {safe(row['context'])} |")
                lines.extend(["", "Human interpretation: _pending review._", ""])

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            json.dump(result, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
        with args.markdown.open("x", encoding="utf-8") as target:
            target.write("\n".join(lines))
        print(f"Review JSON: {args.output}; worksheet: {args.markdown}")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
