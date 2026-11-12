"""Extract entities with GLiNER from natural-language type descriptions, with no training."""

import argparse
import time

from gliner import GLiNER

from models.llm.common import (
    PROJECT, RESULTS, TYPE_DESCRIPTIONS, char_spans_to_tags, load_input, load_rows, per_type_scores,
    save_json, score_rows, subset_path, write_predictions,
)


MODEL = "urchade/gliner_medium-v2.1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=list(TYPE_DESCRIPTIONS), choices=list(TYPE_DESCRIPTIONS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--subset", default="test_300", help="ID list name under data/subsets, or 'full'")
    parser.add_argument("--input", help="Derived test file to label instead of a split; needs --scope and one dataset")
    parser.add_argument("--scope", help="Name of the --input test set in method names and reports")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = GLiNER.from_pretrained(MODEL).to(args.device).eval()
    if args.input and (not args.scope or len(args.dataset) != 1):
        parser.error("--input needs --scope and exactly one --dataset")
    subset = None if args.subset == "full" else args.subset
    scope = args.scope or args.subset
    for dataset in args.dataset:
        rows = load_input(args.input) if args.input else load_rows(dataset, args.split, subset)
        descriptions = TYPE_DESCRIPTIONS[dataset]
        by_description = {text: kind for kind, text in descriptions.items()}
        texts = [" ".join(row["tokens"]) for row in rows]
        start = time.perf_counter()
        outputs = model.batch_predict_entities(
            texts, list(by_description), threshold=args.threshold, batch_size=args.batch_size,
        )
        seconds = time.perf_counter() - start
        tags = {
            row["sample_id"]: char_spans_to_tags(
                row["tokens"], [(item["start"], item["end"], by_description[item["label"]]) for item in found],
            )
            for row, found in zip(rows, outputs, strict=True)
        }
        method = f"gliner_{scope}"
        path = write_predictions(dataset, method, rows, tags)
        report = {
            "method": "GLiNER", "model": MODEL, "dataset": dataset, "split": args.split, "scope": scope,
            "input": args.input,
            "subset_ids": (None if subset is None or args.input
                           else subset_path(dataset, subset).relative_to(PROJECT).as_posix()),
            "threshold": args.threshold, "type_descriptions": descriptions,
            "predictions": path.relative_to(PROJECT).as_posix(),
            "scores": score_rows(rows, tags),
            "per_type": per_type_scores(rows, tags),
            "timing": {"device": args.device, "batch_size": args.batch_size, "inference_seconds": seconds,
                       "inference_sentences_per_second": len(rows) / seconds,
                       "includes": "tokenization and span decoding"},
        }
        save_json(RESULTS / f"gliner_{dataset}_{args.split}_{scope}.json", report)
        print(f"{dataset} {scope}: F1={report['scores']['f1']:.2%} "
              f"P={report['scores']['precision']:.2%} R={report['scores']['recall']:.2%}", flush=True)


if __name__ == "__main__":
    main()
