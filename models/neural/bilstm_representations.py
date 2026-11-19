"""Compare four inputs with the same BiLSTM-CRF architecture on validation.

Static 100-dimensional inputs are zero-padded to BERT's 768 dimensions. The
additional 30 character dimensions are zero except in glove_char mode. Random
and GloVe embeddings are frozen, as is the pretrained BERT encoder; the LSTM,
classification layer, and CRF always have the same shape and optimizer settings.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torchcrf import CRF
from transformers import AutoModel, AutoTokenizer
from models.neural.token_alignment import tokenize_with_unknown_fallback

from eval.score_ner import read_subset_ids, score_ner
from models.llm.common import TYPE_DESCRIPTIONS, load_rows, sample_train
from models.neural.bert_baseline import fix_iob2
from models.neural.bilstm_crf import GLOVE_PATH, load_glove, make_character_vocab


BERT_NAME = "bert-base-cased"
BERT_REVISION = "cd5ef92a9fb2f889e972770a36d4ed042daf221e"
WORD_DIM = 768
CHAR_DIM = 30


def select_device(choice):
    if choice == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(choice)


def fingerprint(rows):
    """Identify exactly the token input without using gold labels."""
    digest = hashlib.sha256()
    digest.update(f"{BERT_NAME}@{BERT_REVISION}\n".encode())
    for row in rows:
        digest.update(json.dumps([row["sample_id"], row["tokens"]], ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def read_cached_features(path, digest, rows):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("fingerprint") != digest or saved.get("model") != BERT_NAME:
        raise ValueError(f"{path}: cached frozen features do not match the token input")
    features = saved.get("features")
    if not isinstance(features, list) or len(features) != len(rows):
        raise ValueError(f"{path}: invalid frozen feature count")
    for row, tensor in zip(rows, features, strict=True):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (len(row["tokens"]), WORD_DIM):
            raise ValueError(f'{path}: invalid frozen feature shape for {row["sample_id"]}')
    return features


def encode_frozen(rows, tokenizer, encoder, device, batch_size):
    """First-subword representations from frozen BERT; only input tokens are read."""
    features = []
    encoder.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            _, encoding, affected = tokenize_with_unknown_fallback(batch, tokenizer)
            if affected:
                print(json.dumps({"token_alignment_fallback": affected}, ensure_ascii=False), flush=True)
            if encoding["input_ids"].shape[1] > encoder.config.max_position_embeddings:
                raise ValueError("Frozen BERT input exceeds maximum length; refusing truncation")
            positions = []
            for i, row in enumerate(batch):
                first = []
                previous = None
                for pos, word_id in enumerate(encoding.word_ids(batch_index=i)):
                    if word_id is not None and word_id != previous:
                        first.append(pos)
                    previous = word_id
                if len(first) != len(row["tokens"]):
                    raise ValueError(f'{row["sample_id"]}: word/subword alignment failed')
                positions.append(first)
            hidden = encoder(**encoding.to(device)).last_hidden_state
            for i, indices in enumerate(positions):
                features.append(hidden[i, indices].detach().to("cpu", dtype=torch.float32))
            if min(offset + batch_size, len(rows)) % 400 == 0 or offset + batch_size >= len(rows):
                print(f"Frozen BERT features: {min(offset + batch_size, len(rows))}/{len(rows)}", flush=True)
    return features


def features_for_sets(splits, device, batch_size, cache_dir):
    found = {}
    missing = []
    for split, rows in splits:
        digest = fingerprint(rows)
        cache_path = Path(cache_dir) / f"bert_{split}_{digest[:20]}.pt" if cache_dir else None
        if cache_path is not None and cache_path.exists():
            print(f"Loading frozen BERT cache: {cache_path}", flush=True)
            found[split] = read_cached_features(cache_path, digest, rows)
        else:
            missing.append((split, rows, digest, cache_path))
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(
            BERT_NAME, revision=BERT_REVISION, local_files_only=True, use_fast=True,
        )
        encoder = AutoModel.from_pretrained(
            BERT_NAME, revision=BERT_REVISION, local_files_only=True,
        ).to(device)
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        for split, rows, digest, cache_path in missing:
            print(f"Computing frozen BERT features for {split}...", flush=True)
            values = encode_frozen(rows, tokenizer, encoder, device, batch_size)
            found[split] = values
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                # Publish only a complete cache and never overwrite a concurrent job.
                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(dir=cache_path.parent, prefix=".bert_features_", delete=False) as file:
                        temp_path = Path(file.name)
                        torch.save({"fingerprint": digest, "model": BERT_NAME,
                                    "revision": BERT_REVISION, "features": values}, file)
                    try:
                        os.link(temp_path, cache_path)
                    except FileExistsError:
                        read_cached_features(cache_path, digest, rows)
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
                print(f"Saved frozen BERT cache: {cache_path}", flush=True)
        del encoder
        if device.type == "mps":
            torch.mps.empty_cache()
    return found


def batch_tensors(rows, word_to_id, char_to_id, label_to_id, device, mode):
    batch_size = len(rows)
    max_words = max(len(row["tokens"]) for row in rows)
    max_chars = max(len(token) for row in rows for token in row["tokens"])
    word_ids = torch.zeros((batch_size, max_words), dtype=torch.long)
    char_ids = torch.zeros((batch_size, max_words, max_chars), dtype=torch.long)
    tag_ids = torch.zeros((batch_size, max_words), dtype=torch.long)
    mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
    lengths = torch.tensor([len(row["tokens"]) for row in rows], dtype=torch.long)
    frozen_features = torch.zeros((batch_size, max_words, WORD_DIM)) if mode == "bert" else None
    for batch_id, row in enumerate(rows):
        tokens, tags = row["tokens"], row["tags"]
        if not tokens or len(tokens) != len(tags):
            raise ValueError(f'{row["sample_id"]}: token/tag mismatch')
        if mode == "bert":
            frozen_features[batch_id, :len(tokens)] = row["_bert"]
        for index, (token, tag) in enumerate(zip(tokens, tags, strict=True)):
            if mode != "bert":
                word_ids[batch_id, index] = word_to_id.get(token.lower(), word_to_id["<unk>"])
            tag_ids[batch_id, index] = label_to_id[tag]
            mask[batch_id, index] = True
            if mode == "glove_char":
                for pos, char in enumerate(token):
                    char_ids[batch_id, index, pos] = char_to_id.get(char, char_to_id["<unk>"])
    return (word_ids.to(device), char_ids.to(device), tag_ids.to(device), mask.to(device),
            lengths, frozen_features.to(device) if frozen_features is not None else None)


class RepresentationTagger(nn.Module):
    def __init__(self, mode, vectors, num_chars, num_labels, hidden_size, dropout):
        super().__init__()
        self.mode = mode
        self.word_embedding = None if vectors is None else nn.Embedding.from_pretrained(
            vectors, freeze=True, padding_idx=0,
        )
        self.char_embedding = nn.Embedding(num_chars, 30, padding_idx=0)
        self.char_cnn = nn.Conv1d(30, CHAR_DIM, kernel_size=3, padding=1)
        if mode != "glove_char":
            for parameter in list(self.char_embedding.parameters()) + list(self.char_cnn.parameters()):
                parameter.requires_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(WORD_DIM + CHAR_DIM, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def emissions(self, word_ids, char_ids, lengths, bert_features):
        batch_size, max_words, max_chars = char_ids.shape
        if self.mode == "bert":
            word_features = bert_features
        else:
            static = self.word_embedding(word_ids)
            # Same downstream architecture for all four inputs; no learned projection.
            word_features = torch.nn.functional.pad(static, (0, WORD_DIM - static.shape[-1]))
        if self.mode == "glove_char":
            flat = char_ids.reshape(batch_size * max_words, max_chars)
            valid = flat.ne(0)
            chars = self.char_embedding(flat).transpose(1, 2)
            chars = torch.relu(self.char_cnn(chars)).masked_fill(~valid.unsqueeze(1), -1e4)
            chars = chars.max(dim=-1).values
            chars = torch.where(valid.any(dim=-1, keepdim=True), chars, torch.zeros_like(chars))
            char_features = chars.reshape(batch_size, max_words, CHAR_DIM)
        else:
            char_features = word_features.new_zeros((batch_size, max_words, CHAR_DIM))
        inputs = self.dropout(torch.cat([word_features, char_features], dim=-1))
        packed = pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bilstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=max_words)
        return self.classifier(self.dropout(output))


def evaluate(model, loader, words, chars, labels, inverse, device, mode):
    model.eval()
    gold, predictions = {}, {}
    with torch.no_grad():
        for rows in loader:
            word_ids, char_ids, _, mask, lengths, features = batch_tensors(rows, words, chars, labels, device, mode)
            emissions = model.emissions(word_ids, char_ids, lengths, features)
            decoded = model.crf.decode(emissions, mask=mask)
            for row, sequence in zip(rows, decoded, strict=True):
                if len(sequence) != len(row["tokens"]):
                    raise ValueError(f'{row["sample_id"]}: decoded length mismatch')
                gold[row["sample_id"]] = row["tags"]
                predictions[row["sample_id"]] = fix_iob2([inverse[index] for index in sequence])
    return score_ner(gold, predictions), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("random", "glove", "glove_char", "bert"))
    parser.add_argument("--dataset", default="conll2003", choices=("conll2003", "wnut17", "mit_restaurant"))
    parser.add_argument("--train-size", default="1.0")
    parser.add_argument("--dev-size", type=int, default=0, help="0 uses all validation sentences")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bert-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--bert-cache-dir", help="Optional directory for reusable token-only frozen features")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.bert_batch_size < 1 or args.dev_size < 0 or args.hidden_size < 1 or not 0 <= args.dropout < 1:
        parser.error("invalid training arguments")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    train = sample_train(args.dataset, args.train_size, args.seed)
    dev = load_rows(args.dataset, "validation")
    if args.dev_size:
        dev = dev[:args.dev_size]
    if not train or not dev:
        raise ValueError("Training and validation sets must be nonempty")
    chars = make_character_vocab(train)
    labels = ["O"]
    for entity_type in sorted(TYPE_DESCRIPTIONS[args.dataset]):
        labels.extend((f"B-{entity_type}", f"I-{entity_type}"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    inverse = {index: label for label, index in label_to_id.items()}
    words, vectors, glove_sha256, random_scale = None, None, None, None
    print(f"Mode: {args.mode}; device: {device}; train={len(train)}; validation={len(dev)}", flush=True)
    if args.mode == "bert":
        features = features_for_sets((("train", train), ("validation", dev)), device,
                                     args.bert_batch_size, args.bert_cache_dir)
        train_features, dev_features = features["train"], features["validation"]
        train = [{**row, "_bert": vector} for row, vector in zip(train, train_features, strict=True)]
        dev = [{**row, "_bert": vector} for row, vector in zip(dev, dev_features, strict=True)]
    else:
        words, glove, glove_sha256 = load_glove(GLOVE_PATH)
        if args.mode == "random":
            # Match GloVe's overall scale, but do not preserve its word semantics.
            generator = torch.Generator().manual_seed(args.seed + 100000)
            random_scale = glove[1:].std().item()
            vectors = torch.randn(glove.shape, generator=generator) * random_scale
            vectors[0].zero_()
            del glove
        else:
            vectors = glove
    # Keep trainable model initialization and shuffle independent of feature setup.
    torch.manual_seed(args.seed)
    model = RepresentationTagger(args.mode, vectors, len(chars), len(labels), args.hidden_size, args.dropout).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              collate_fn=list, generator=shuffle_generator)
    dev_loader = DataLoader(dev, batch_size=args.batch_size, shuffle=False, collate_fn=list)
    output.mkdir(parents=True)
    config = vars(args).copy()
    config.update(device=str(device), labels=labels, glove_sha256=glove_sha256,
                  static_embeddings_frozen=True if args.mode != "bert" else None,
                  bert_frozen=True if args.mode == "bert" else None,
                  bert_model=BERT_NAME if args.mode == "bert" else None,
                  bert_revision=BERT_REVISION if args.mode == "bert" else None,
                  static_padding_to=WORD_DIM, character_dimensions=CHAR_DIM,
                  lstm_input_dimensions=WORD_DIM + CHAR_DIM,
                  random_initialization="normal_scaled_by_glove_std" if args.mode == "random" else None,
                  random_initialization_std=random_scale,
                  torch_version=str(torch.__version__), numpy_version=str(np.__version__))
    with (output / "run_config.json").open("x", encoding="utf-8") as target:
        json.dump(config, target, ensure_ascii=False, indent=2)
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started, loss_sum = time.perf_counter(), 0.0
        for number, rows in enumerate(train_loader, 1):
            word_ids, char_ids, targets, mask, lengths, features = batch_tensors(
                rows, words, chars, label_to_id, device, args.mode,
            )
            optimizer.zero_grad()
            emissions = model.emissions(word_ids, char_ids, lengths, features)
            loss = -model.crf(emissions, targets, mask=mask, reduction="token_mean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, foreach=False)
            optimizer.step()
            loss_sum += loss.item()
            if number == 1 or number % 100 == 0:
                print(f"Epoch {epoch}/{args.epochs} | batch {number}/{len(train_loader)} | loss {loss.item():.4f}", flush=True)
        scores, predictions = evaluate(model, dev_loader, words, chars, label_to_id, inverse, device, args.mode)
        print(f"Epoch {epoch} | seconds={time.perf_counter()-started:.1f} | mean_loss={loss_sum/len(train_loader):.4f}", flush=True)
        print(json.dumps(scores, indent=2), flush=True)
        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]
            torch.save({"model_state_dict": model.state_dict(), "word_to_id": words, "char_to_id": chars,
                        "label_to_id": label_to_id, "config": config, "best_epoch": epoch}, output / "best_model.pt")
            with (output / "validation_scores.json").open("w", encoding="utf-8") as target:
                json.dump({**scores, "best_epoch": epoch}, target, ensure_ascii=False, indent=2)
            with (output / "validation_predictions.jsonl").open("w", encoding="utf-8") as target:
                for sample_id, tags in predictions.items():
                    target.write(json.dumps({"sample_id": sample_id, "tags": tags}, ensure_ascii=False) + "\n")
            print(f"Best checkpoint saved at epoch {epoch}", flush=True)
    print("Done.", flush=True)


def predict_main():
    parser = argparse.ArgumentParser(description="Predict from a saved input-representation checkpoint.")
    parser.add_argument("--model", required=True, help="Checkpoint .pt or directory containing best_model.pt")
    parser.add_argument("--input", required=True, help="JSONL with sample_id and tokens; gold tags are ignored")
    parser.add_argument("--output", required=True, help="New prediction JSONL path")
    parser.add_argument("--subset-ids", help="Optional file listing exactly the sample IDs to predict")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bert-batch-size", type=int, default=8)
    parser.add_argument("--bert-cache-dir", help="Optional frozen feature cache directory")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args(sys.argv[2:])
    if args.batch_size < 1 or args.bert_batch_size < 1:
        parser.error("batch sizes must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    rows, by_id = [], {}
    with Path(args.input).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id, tokens = row.get("sample_id"), row.get("tokens")
            if not isinstance(sample_id, str) or not sample_id or sample_id in by_id:
                raise ValueError(f"{args.input}:{line_number}: missing or duplicate sample_id")
            if not isinstance(tokens, list) or not tokens or any(not isinstance(t, str) or not t.strip() for t in tokens):
                raise ValueError(f"{args.input}:{line_number}: invalid tokens")
            clean = {"sample_id": sample_id, "tokens": tokens, "tags": ["O"] * len(tokens)}
            rows.append(clean)
            by_id[sample_id] = clean
    if args.subset_ids:
        ids = read_subset_ids(Path(args.subset_ids))
        missing = [sample_id for sample_id in ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} subset IDs are missing, e.g. {missing[0]}")
        rows = [by_id[sample_id] for sample_id in ids]
    if not rows:
        raise ValueError("No sentences to predict")
    model_path = Path(args.model)
    if model_path.is_dir():
        model_path = model_path / "best_model.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    config, state = checkpoint["config"], checkpoint["model_state_dict"]
    mode = config["mode"]
    if mode not in ("random", "glove", "glove_char", "bert"):
        raise ValueError(f"Unsupported representation mode: {mode}")
    if config["lstm_input_dimensions"] != WORD_DIM + CHAR_DIM:
        raise ValueError("Checkpoint uses a different LSTM input architecture")
    device = select_device(args.device)
    words, chars, labels = checkpoint["word_to_id"], checkpoint["char_to_id"], checkpoint["label_to_id"]
    inverse = {index: label for label, index in labels.items()}
    if mode == "bert":
        if config.get("bert_revision") != BERT_REVISION:
            raise ValueError("Checkpoint uses a different frozen BERT revision")
        feature_sets = features_for_sets((("prediction", rows),), device, args.bert_batch_size, args.bert_cache_dir)
        rows = [{**row, "_bert": vector} for row, vector in zip(rows, feature_sets["prediction"], strict=True)]
        vectors = None
    else:
        vectors = state["word_embedding.weight"]
    model = RepresentationTagger(mode, vectors, len(chars), len(labels), config["hidden_size"], config["dropout"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print(f"Predicting {len(rows)} sentences with {mode} on {device}", flush=True)
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            word_ids, char_ids, _, mask, lengths, features = batch_tensors(batch, words, chars, labels, device, mode)
            sequences = model.crf.decode(model.emissions(word_ids, char_ids, lengths, features), mask=mask)
            for row, sequence in zip(batch, sequences, strict=True):
                if len(sequence) != len(row["tokens"]):
                    raise ValueError(f'{row["sample_id"]}: decoded length mismatch')
                predictions.append({"sample_id": row["sample_id"],
                                    "tags": fix_iob2([inverse[index] for index in sequence])})
            completed = min(offset + args.batch_size, len(rows))
            if completed == len(rows) or completed % 400 == 0:
                print(f"Predicted {completed}/{len(rows)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        for prediction in predictions:
            target.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(f"Saved {len(predictions)} predictions to {output}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        predict_main()
    else:
        main()
