"""Predict word-level BIO tags for a JSONL file with a saved BERT, BiLSTM-CRF or span checkpoint."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer, BertConfig

from eval.score_ner import read_subset_ids
from models.neural.bert_baseline import fix_iob2
from models.neural.bilstm_crf import BiLSTMCRF, encode_batch
from models.neural.span_bert import prepare_batch, spans_to_bio
from models.neural.token_alignment import tokenize_with_unknown_fallback


class RestoredSpan(nn.Module):
    """Recreate the saved Span architecture without loading pretrained weights."""

    def __init__(self, state, config):
        super().__init__()
        model_name = config.get("model_name")
        if model_name != "bert-base-cased":
            raise ValueError(f"Unsupported Span encoder: {model_name!r}")
        try:
            encoder_config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        except OSError:
            # The checkpoint does not store the encoder config; without a cached copy,
            # rebuild the standard bert-base-cased architecture from the weight shapes.
            vocab_size, hidden_size = state["encoder.embeddings.word_embeddings.weight"].shape
            if hidden_size != 768:
                raise ValueError("Cannot reconstruct a nonstandard BERT encoder without its config")
            encoder_config = BertConfig(vocab_size=vocab_size)
        self.encoder = AutoModel.from_config(encoder_config)
        length_count, length_size = state["length_embedding.weight"].shape
        self.length_embedding = nn.Embedding(length_count, length_size)
        hidden_units, feature_size = state["classifier.0.weight"].shape
        class_count = state["classifier.3.weight"].shape[0]
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_units), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_units, class_count),
        )

    def forward(self, encoding, batch_ids, starts, ends, lengths):
        hidden = self.encoder(**encoding).last_hidden_state
        features = torch.cat(
            [hidden[batch_ids, starts], hidden[batch_ids, ends], self.length_embedding(lengths)],
            dim=-1,
        )
        return self.classifier(features)


def input_rows(path, subset):
    """Read sample_id and tokens from JSONL, optionally restricted to and ordered by a subset file."""
    rows = []
    by_id = {}
    with Path(path).open(encoding="utf-8-sig") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id, tokens = row.get("sample_id"), row.get("tokens")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"{path}:{lineno}: missing sample_id")
            if sample_id in by_id:
                raise ValueError(f"{path}:{lineno}: duplicate sample_id {sample_id!r}")
            if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t.strip() for t in tokens):
                raise ValueError(f"{path}:{lineno}: tokens must be a nonempty list of nonempty strings")
            clean = {"sample_id": sample_id, "tokens": tokens}
            rows.append(clean)
            by_id[sample_id] = clean
    if subset:
        ids = read_subset_ids(Path(subset))
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} subset IDs missing from input, e.g. {missing[0]}")
        rows = [by_id[sample_id] for sample_id in ids]
    if not rows:
        raise ValueError("No sentences to predict")
    return rows


def choose_device(name):
    """Resolve auto to cuda, then mps, then cpu; reject an unavailable device."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def restore_bilstm(checkpoint, device):
    state = checkpoint["model_state_dict"]
    words, chars, labels = (checkpoint[key] for key in ("word_to_id", "char_to_id", "label_to_id"))
    config = checkpoint["config"]
    model = BiLSTMCRF(
        glove_vectors=state["word_embedding.weight"],
        num_characters=len(chars), num_labels=len(labels),
        hidden_size=int(config["hidden_size"]), dropout=float(config["dropout"]),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), words, chars, labels


def restore_bert(model_path, device):
    """Load the saved token classifier and its tokenizer from a local directory."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
    return model.to(device).eval(), tokenizer


def bert_predict(model, rows, tokenizer, batch_size, device):
    """Tag rows with the token classifier, reading each word's label at its first subword."""
    inverse = {int(index): label for index, label in model.config.id2label.items()}
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            _, encoding, affected = tokenize_with_unknown_fallback(batch, tokenizer)
            if affected:
                print(json.dumps({"token_alignment_fallback": affected}, ensure_ascii=False), flush=True)
            if encoding["input_ids"].shape[1] > model.config.max_position_embeddings:
                raise ValueError("A tokenized sentence exceeds the BERT limit; refusing truncation")
            positions = []
            for i, row in enumerate(batch):
                first = []
                previous = None
                for index, word_id in enumerate(encoding.word_ids(batch_index=i)):
                    if word_id is not None and word_id != previous:
                        first.append(index)
                    previous = word_id
                if len(first) != len(row["tokens"]):
                    raise ValueError(f'{row["sample_id"]}: word/subword alignment failed')
                positions.append(first)
            encoding = encoding.to(device)
            logits = model(**encoding).logits
            predicted = logits.argmax(dim=-1).cpu().tolist()
            for i, row in enumerate(batch):
                tags = fix_iob2([inverse[predicted[i][position]] for position in positions[i]])
                results.append({"sample_id": row["sample_id"], "tags": tags})
            completed = min(start + batch_size, len(rows))
            if completed == len(rows) or completed % 400 == 0:
                print(f"Predicted {completed}/{len(rows)}", flush=True)
    return results


def bilstm_predict(model, rows, state, batch_size, device):
    """Tag rows with the BiLSTM-CRF using Viterbi decoding."""
    words, chars, labels = state
    inverse = {index: tag for tag, index in labels.items()}
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            # encode_batch requires tags; placeholders suffice because the tag ids are discarded.
            encoded_rows = [{**row, "tags": ["O"] * len(row["tokens"])} for row in batch]
            word_ids, char_ids, _, mask, lengths = encode_batch(encoded_rows, words, chars, labels, device)
            emissions = model.emissions(word_ids, char_ids, lengths)
            sequences = model.crf.decode(emissions, mask=mask)
            for row, sequence in zip(batch, sequences, strict=True):
                if len(sequence) != len(row["tokens"]):
                    raise ValueError(f'{row["sample_id"]}: decoded length mismatch')
                results.append({"sample_id": row["sample_id"], "tags": fix_iob2([inverse[i] for i in sequence])})
            completed = min(start + batch_size, len(rows))
            if completed == len(rows) or completed % 400 == 0:
                print(f"Predicted {completed}/{len(rows)}", flush=True)
    return results


def restore_span(checkpoint, model_path, tokenizer_path, device):
    """Rebuild the span classifier and load its tokenizer from the directory beside the checkpoint."""
    config, state = checkpoint["config"], checkpoint["state_dict"]
    labels = checkpoint["label_to_id"]
    if set(labels.values()) != set(range(1, len(labels) + 1)):
        raise ValueError("Unexpected Span label IDs")
    tokenizer_dir = Path(tokenizer_path) if tokenizer_path else model_path.parent / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True, local_files_only=True)
    model = RestoredSpan(state, config)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), tokenizer, labels, config


def span_predict(model, rows, state, batch_size, device):
    """Tag rows with the span classifier, resolving overlapping spans by confidence."""
    tokenizer, labels, config = state
    inverse = {index: label for label, index in labels.items()}
    results = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            corrected, _, affected = tokenize_with_unknown_fallback(batch, tokenizer)
            if affected:
                print(json.dumps({"token_alignment_fallback": affected}, ensure_ascii=False), flush=True)
            encoding, tensors, _, metadata = prepare_batch(
                corrected, tokenizer, labels, int(config["max_span_length"]),
                int(config["negative_ratio"]), None, device, False,
            )
            logits = model(encoding, **tensors)
            probabilities = torch.softmax(logits, dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            confidence, predicted = confidence.cpu().tolist(), predicted.cpu().tolist()
            proposals = [[] for _ in batch]
            for i, (batch_id, left, right) in enumerate(metadata):
                label_id = predicted[i]
                if label_id:
                    proposals[batch_id].append((confidence[i], left, right, inverse[label_id]))
            for i, row in enumerate(batch):
                results.append({"sample_id": row["sample_id"], "tags": spans_to_bio(len(row["tokens"]), proposals[i])})
            completed = min(start + batch_size, len(rows))
            if completed == len(rows) or completed % 400 == 0:
                print(f"Predicted {completed}/{len(rows)}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("bert", "bilstm", "span"))
    parser.add_argument("--model", required=True,
                        help="Model training output directory, or its checkpoint/model directory")
    parser.add_argument("--input", required=True, help="Input JSONL with sample_id and tokens")
    parser.add_argument("--output", required=True, help="Prediction JSONL; must not already exist")
    parser.add_argument("--subset-ids", help="Optional file with one sample_id per line")
    parser.add_argument("--tokenizer", help="Span tokenizer directory; defaults to sibling tokenizer/")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    model_path = Path(args.model)
    if args.method == "bert" and model_path.is_dir() and not (model_path / "config.json").is_file():
        model_path = model_path / "best_model"
    elif args.method != "bert" and model_path.is_dir():
        model_path = model_path / "best_model.pt"
    rows = input_rows(args.input, args.subset_ids)
    device = choose_device(args.device)
    print(f"Device: {device}", flush=True)
    print(f"Loading {args.method} from {model_path} on {device}; sentences={len(rows)}", flush=True)
    if args.method == "bert":
        model, tokenizer = restore_bert(model_path, device)
        state = tokenizer
        predictor = bert_predict
    elif args.method == "bilstm":
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model, words, chars, labels = restore_bilstm(checkpoint, device)
        state = (words, chars, labels)
        predictor = bilstm_predict
    else:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model, tokenizer, labels, config = restore_span(checkpoint, model_path, args.tokenizer, device)
        state = (tokenizer, labels, config)
        predictor = span_predict
    predictions = predictor(model, rows, state, args.batch_size, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for prediction in predictions:
            target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions to {output}")


if __name__ == "__main__":
    main()
