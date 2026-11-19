"""Make F4's matched GloVe/base-BERT/fine-tuned-BERT t-SNE figure.

All panels use the identical intersection of gold span IDs for which each
representation is available. K-means and ARI/NMI use the original vectors,
not the two-dimensional t-SNE coordinates. No supervised model is fitted.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from analysis import style
from analysis.cluster_entities import glove_entity_vectors, read_bert_vectors, read_gold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--glove", type=Path, required=True)
    parser.add_argument("--base-bert", type=Path, required=True)
    parser.add_argument("--fine-tuned-bert", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Vector PDF output")
    parser.add_argument("--metrics", type=Path, required=True, help="ARI/NMI metadata JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional common deterministic subset; 0 uses all matched spans")
    args = parser.parse_args()
    try:
        if args.output.suffix.lower() != ".pdf":
            raise ValueError("--output must have a .pdf extension")
        inputs = (args.gold, args.glove, args.base_bert, args.fine_tuned_bert)
        if args.output.exists() or args.metrics.exists():
            raise FileExistsError("figure and metrics output paths must be new")
        if len({args.output.resolve(), args.metrics.resolve(), *(p.resolve() for p in inputs)}) != 6:
            raise ValueError("input and output paths must be distinct")
        if args.max_samples < 0 or args.perplexity <= 0:
            raise ValueError("max-samples must be nonnegative and perplexity positive")

        print("Reading matching gold spans and representations...", flush=True)
        gold = read_gold(args.gold)
        representations = {
            "GloVe": glove_entity_vectors(args.glove, gold),
            "BERT (base)": read_bert_vectors(args.base_bert, gold),
            "BERT (fine-tuned)": read_bert_vectors(args.fine_tuned_bert, gold),
        }
        keys = sorted(set.intersection(*(set(vectors) for vectors, _ in representations.values())))
        all_matched = len(keys)
        if args.max_samples and len(keys) > args.max_samples:
            rng = np.random.default_rng(args.seed)
            selected = sorted(rng.choice(len(keys), size=args.max_samples, replace=False).tolist())
            keys = [keys[index] for index in selected]
        types = [gold[key][0] for key in keys]
        if set(types) != set(style.ENTITY) or len(keys) < 5 or args.perplexity >= len(keys):
            raise ValueError("selected spans must cover all 4 types and exceed the t-SNE perplexity")

        style.setup()
        fig, axes = plt.subplots(1, 3, figsize=(style.TEXT_WIDTH, 2.5), constrained_layout=True)
        report = {
            "gold": str(args.gold), "representations": {}, "matched_before_sampling": all_matched,
            "used_spans": len(keys), "seed": args.seed, "perplexity": args.perplexity,
            "max_samples": args.max_samples,
            "matching": "identical gold (sample_id,start,end) intersection across all 3 representations",
            "clustering": "K-means k=4, n_init=10 on original vectors without normalization",
            "embedding": "t-SNE on original vectors, per representation; axes have arbitrary units",
        }
        for axis, (name, (vectors, metadata)) in zip(axes, representations.items(), strict=True):
            matrix = np.stack([vectors[key] for key in keys])
            print(f"Computing {name} clustering and t-SNE for {len(keys)} spans...", flush=True)
            with threadpool_limits(limits=2):
                clusters = KMeans(n_clusters=4, n_init=10, random_state=args.seed).fit_predict(matrix)
                points = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                              learning_rate="auto", random_state=args.seed).fit_transform(matrix)
            ari = float(adjusted_rand_score(types, clusters))
            nmi = float(normalized_mutual_info_score(types, clusters))
            report["representations"][name] = {"ari": ari, "nmi": nmi, "source": metadata}
            for kind, color in style.ENTITY.items():
                indices = np.asarray([label == kind for label in types])
                axis.scatter(points[indices, 0], points[indices, 1], s=3, alpha=0.5,
                             color=color, label=kind, linewidths=0)
            axis.set_title(name)
            axis.text(0.02, 0.02, f"ARI {ari:.3f}\nNMI {nmi:.3f}",
                      transform=axis.transAxes, fontsize=7, va="bottom")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as temp:
            staged = Path(temp) / args.output.name
            fig.savefig(staged, format="pdf")
            with args.output.open("xb") as target, staged.open("rb") as source:
                target.write(source.read())
        plt.close(fig)
        with args.metrics.open("x", encoding="utf-8") as target:
            json.dump(report, target, ensure_ascii=False, indent=2)
            target.write("\n")
        print(f"Figure: {args.output}; metrics: {args.metrics}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
