"""Export the mean BERT subword vector of every gold entity span to JSONL.

The encoder is a base BERT model or a fine-tuned token-classification checkpoint; span ends are exclusive.
"""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from transformers import AutoModel, AutoModelForTokenClassification, AutoTokenizer

from eval.score_ner import bio_to_spans, read_subset_ids
from models.neural.token_alignment import tokenize_with_unknown_fallback


def read_rows(path, ids_path=None):
    """Read and validate tagged sentences, optionally restricted to and ordered by a subset file."""
    rows = []
    seen = set()
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            if not isinstance(sample_id, str) or not sample_id.strip() or sample_id in seen:
                raise ValueError(f"{path}:{number}: invalid or duplicate sample ID")
            tokens, tags = row["tokens"], row["tags"]
            if not isinstance(tokens, list) or not tokens or len(tokens) != len(tags):
                raise ValueError(f"{sample_id}: expected nonempty, aligned tokens and tags")
            bio_to_spans(tags)
            seen.add(sample_id)
            rows.append(row)
    if ids_path:
        ids = read_subset_ids(ids_path)
        by_id = {row["sample_id"]: row for row in rows}
        missing = set(ids) - by_id.keys()
        if missing:
            raise ValueError(f"subset ID {min(missing)!r} absent from {path}")
        rows = [by_id[sample_id] for sample_id in ids]
    if not rows:
        raise ValueError("input has no sentences")
    return rows


def load_encoder(model_path, fine_tuned, offline):
    """Load a BERT encoder, taking it from the token classifier when the model is fine-tuned."""
    if fine_tuned:
        # Loading the same class as training makes the encoder weights explicit.
        classifier = AutoModelForTokenClassification.from_pretrained(
            model_path, local_files_only=offline
        )
        return classifier.base_model
    return AutoModel.from_pretrained(model_path, local_files_only=offline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL with sample_id, tokens and gold tags")
    parser.add_argument("--model", required=True, help="Base model ID or local token-classification checkpoint")
    parser.add_argument("--fine-tuned", action="store_true", help="Model is a token-classification checkpoint")
    parser.add_argument("--subset-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--offline", action="store_true", help="Use cached or local files only")
    args = parser.parse_args()

    try:
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        if args.output.exists():
            raise FileExistsError(f"{args.output} already exists")
        if args.output.resolve() in {p.resolve() for p in (args.input, args.subset_ids) if p}:
            raise ValueError("output must not overwrite an input")
        rows = read_rows(args.input, args.subset_ids)
        choice = args.device
        if choice == "auto":
            choice = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        if choice == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available")
        if choice == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        device = torch.device(choice)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model, use_fast=True, local_files_only=args.offline
        )
        if not tokenizer.is_fast:
            raise ValueError("a fast tokenizer with word_ids support is required")
        model = load_encoder(args.model, args.fine_tuned, args.offline).to(device)
        model.eval()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        spans_written = 0
        print(f"Device: {device}; sentences: {len(rows)}; model: {args.model}", flush=True)

        with TemporaryDirectory(dir=args.output.parent) as tmp, torch.no_grad():
            staged = Path(tmp) / args.output.name
            with staged.open("x", encoding="utf-8", newline="\n") as output:
                for offset in range(0, len(rows), args.batch_size):
                    batch = rows[offset:offset + args.batch_size]
                    _, encoding, affected = tokenize_with_unknown_fallback(batch, tokenizer)
                    if affected:
                        print(json.dumps({"token_alignment_fallback": affected}, ensure_ascii=False), flush=True)
                    if encoding["input_ids"].shape[1] > model.config.max_position_embeddings:
                        raise ValueError("sentence exceeds BERT's maximum subword length")
                    word_id_lists = [encoding.word_ids(batch_index=i) for i in range(len(batch))]
                    for row, word_ids in zip(batch, word_id_lists, strict=True):
                        if set(i for i in word_ids if i is not None) != set(range(len(row["tokens"]))):
                            raise ValueError(f"{row['sample_id']}: original words do not align with subwords")
                    hidden = model(**encoding.to(device)).last_hidden_state
                    for row, word_ids, representations in zip(batch, word_id_lists, hidden, strict=True):
                        for start, end, kind in sorted(bio_to_spans(row["tags"])):
                            positions = [index for index, word_id in enumerate(word_ids)
                                         if word_id is not None and start <= word_id < end]
                            if not positions:
                                raise ValueError(f"{row['sample_id']}: entity has no subwords")
                            vector = representations[positions].mean(dim=0).cpu().tolist()
                            output.write(json.dumps({"sample_id": row["sample_id"], "start": start,
                                                     "end": end, "entity_type": kind, "vector": vector},
                                                    ensure_ascii=False, allow_nan=False) + "\n")
                            spans_written += 1
                    done = min(offset + args.batch_size, len(rows))
                    if done % 400 == 0 or done == len(rows):
                        print(f"Processed {done}/{len(rows)} sentences; {spans_written} spans", flush=True)
            # Refuse replacement, even if another process created the result meanwhile.
            with args.output.open("x", encoding="utf-8") as target, staged.open(encoding="utf-8") as source:
                for line in source:
                    target.write(line)
        print(f"Saved {spans_written} entity vectors to {args.output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
