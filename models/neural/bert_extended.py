"""Train BERT taggers for native labels, cross-dataset transfer or pseudo-label distillation."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

from eval.score_ner import bio_to_spans, read_tags_by_id
from eval.score_transfer import MAPPING_PATH, project_tags
from models.llm.common import TYPE_DESCRIPTIONS, load_rows, sample_train
from models.neural.bert_baseline import encode_batch, evaluate


MODEL_NAME = "bert-base-cased"
SOURCES = {
    "conll2003": ("conll2003",),
    "wnut17": ("wnut17",),
    "merged": ("conll2003", "wnut17"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("native", "transfer", "distill"), required=True)
    parser.add_argument("--source", choices=SOURCES, help="Transfer training source; defaults to conll2003")
    parser.add_argument("--dataset", choices=("conll2003", "wnut17", "mit_restaurant"),
                        help="Native-label dataset; defaults to conll2003")
    parser.add_argument("--train-size", default="1.0")
    parser.add_argument("--pseudo-predictions", type=Path)
    parser.add_argument("--label-source", choices=("gold", "pseudo"))
    parser.add_argument("--dev-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def transfer_rows(source, split, size, seed):
    """Load the source datasets with their tags projected to the three shared types."""
    settings = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["transfer_three_types"]
    combined = []
    for dataset in SOURCES[source]:
        rows = sample_train(dataset, size, seed) if split == "train" else load_rows(dataset, split)
        original = {row["sample_id"]: row["tags"] for row in rows}
        if len(original) != len(rows):
            raise ValueError(f"{dataset}/{split}: duplicate sample IDs")
        mapped = project_tags(original, settings["maps"][dataset])
        # Avoid mutating the reusable native-label rows.
        combined.extend({**row, "tags": mapped[row["sample_id"]]} for row in rows)
    return combined


def distillation_rows(prediction_path, label_source):
    """Return the MIT training rows named in the prediction file, with pseudo or gold tags."""
    if prediction_path is None:
        raise ValueError("--pseudo-predictions is required for either distillation condition")
    if label_source not in ("gold", "pseudo"):
        raise ValueError("--label-source gold or pseudo is required for distillation")
    predictions = read_tags_by_id(prediction_path)
    train_rows = load_rows("mit_restaurant", "train")
    by_id = {row["sample_id"]: row for row in train_rows}
    if len(by_id) != len(train_rows):
        raise ValueError("MIT training data contains duplicate sample IDs")
    unknown = predictions.keys() - by_id.keys()
    if unknown:
        raise ValueError(f"Pseudo labels contain non-training IDs, e.g. {min(unknown)}")
    allowed_types = set(TYPE_DESCRIPTIONS["mit_restaurant"])
    for sample_id, tags in predictions.items():
        if len(tags) != len(by_id[sample_id]["tokens"]):
            raise ValueError(f"{sample_id}: pseudo labels and tokens have different lengths")
        spans = bio_to_spans(tags)
        invalid = {kind for _, _, kind in spans} - allowed_types
        if invalid:
            raise ValueError(f"{sample_id}: unexpected pseudo-label types {sorted(invalid)}")
    # Keep corpus order so both label sources train on the same sequence of rows.
    return [
        {**row, "tags": predictions[row["sample_id"]] if label_source == "pseudo" else row["tags"]}
        for row in train_rows if row["sample_id"] in predictions
    ]


def get_device(requested):
    """Resolve auto to cuda, then mps, then cpu; reject an unavailable device."""
    availability = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }
    if requested == "auto":
        requested = "cuda" if availability["cuda"] else "mps" if availability["mps"] else "cpu"
    if not availability[requested]:
        raise ValueError(f"Requested {requested} is not available")
    return torch.device(requested)


def main():
    args = parse_args()
    if min(args.epochs, args.eval_every, args.batch_size) < 1 or args.dev_size < 0:
        raise ValueError("epochs, eval-every, batch-size must be positive and dev-size nonnegative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Output already exists: {args.output_dir}")
    if args.task == "transfer":
        args.source = args.source or "conll2003"
        if args.label_source is not None or args.pseudo_predictions is not None or args.dataset is not None:
            raise ValueError("Pseudo-label/dataset arguments are not available for transfer")
        datasets = SOURCES[args.source]
        train_rows = transfer_rows(args.source, "train", args.train_size, args.seed)
        dev_rows = transfer_rows(args.source, "validation", "1.0", args.seed)
        entity_types = ("LOC", "ORG", "PER")
        validation_policy = "Concatenate source validation splits; no target test data."
    elif args.task == "distill":
        if args.source is not None or args.dataset is not None:
            raise ValueError("--source/--dataset are not available for distillation; it uses MIT Restaurant")
        if args.train_size != "1.0":
            raise ValueError("Distillation must use all IDs from the matched pseudo-prediction file")
        train_rows = distillation_rows(args.pseudo_predictions, args.label_source)
        dev_rows = load_rows("mit_restaurant", "validation")
        entity_types = tuple(sorted(TYPE_DESCRIPTIONS["mit_restaurant"]))
        datasets = ("mit_restaurant",)
        validation_policy = "MIT Restaurant native validation split."
    else:
        if args.source is not None or args.label_source is not None or args.pseudo_predictions is not None:
            raise ValueError("--source is for transfer and pseudo-label arguments are for distillation")
        args.dataset = args.dataset or "conll2003"
        train_rows = sample_train(args.dataset, args.train_size, args.seed)
        dev_rows = load_rows(args.dataset, "validation")
        entity_types = tuple(sorted(TYPE_DESCRIPTIONS[args.dataset]))
        datasets = (args.dataset,)
        validation_policy = f"{args.dataset} native validation split."

    if args.dev_size:
        dev_rows = dev_rows[:args.dev_size]
    if not train_rows or not dev_rows:
        raise ValueError("Training and validation inputs must not be empty")
    train_ids = [row["sample_id"] for row in train_rows]
    dev_ids = [row["sample_id"] for row in dev_rows]
    if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids):
        raise ValueError("Duplicate sample IDs")
    if set(train_ids) & set(dev_ids):
        raise ValueError("Training and validation IDs overlap")
    labels = ["O"]
    for entity_type in entity_types:
        labels.extend((f"B-{entity_type}", f"I-{entity_type}"))
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    for row in train_rows + dev_rows:
        bio_to_spans(row["tags"])
        unexpected = set(row["tags"]) - label_to_id.keys()
        if unexpected:
            raise ValueError(f"{row['sample_id']}: labels not in schema: {sorted(unexpected)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}; training: {len(train_rows)}; validation: {len(dev_rows)}", flush=True)
    print(f"Task: {args.task}; sources: {', '.join(datasets)}; labels: {labels}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, local_files_only=args.local_files_only)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), id2label=id_to_label, label2id=label_to_id,
        local_files_only=args.local_files_only,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=list)
    dev_loader = DataLoader(dev_rows, batch_size=args.batch_size, shuffle=False, collate_fn=list)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "model_name": MODEL_NAME, "device_used": str(device), "source_datasets": datasets,
        "entity_types": entity_types, "labels": labels, "validation_policy": validation_policy,
        "training_sentences": len(train_rows), "validation_sentences": len(dev_rows),
        "selection_metric": "validation_entity_micro_f1",
    })
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    best_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_number, rows in enumerate(train_loader, start=1):
            encoding, aligned, _ = encode_batch(rows, tokenizer, label_to_id, model.config.max_position_embeddings)
            optimizer.zero_grad()
            result = model(**encoding.to(device), labels=aligned.to(device))
            result.loss.backward()
            optimizer.step()
            losses.append(result.loss.item())
            if batch_number == 1 or batch_number % 100 == 0:
                print(f"Epoch {epoch}/{args.epochs} batch {batch_number}/{len(train_loader)} loss {losses[-1]:.4f}", flush=True)
        scores = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            scores, predictions = evaluate(model, dev_loader, tokenizer, label_to_id, id_to_label, device)
            print("Validation scores:", json.dumps(scores), flush=True)
            if scores["f1"] > best_f1:
                best_f1 = scores["f1"]
                model.save_pretrained(args.output_dir / "best_model")
                tokenizer.save_pretrained(args.output_dir / "best_model")
                (args.output_dir / "validation_scores.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
                with (args.output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as file:
                    for sample_id, tags in predictions.items():
                        file.write(json.dumps({"sample_id": sample_id, "tags": tags}, ensure_ascii=False) + "\n")
                (args.output_dir / "best_epoch.json").write_text(json.dumps({"epoch": epoch, "validation_f1": best_f1}) + "\n", encoding="utf-8")
        record = {"epoch": epoch, "mean_training_loss": sum(losses) / len(losses), "validation": scores}
        with (args.output_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        print(f"Epoch {epoch} completed; mean loss {record['mean_training_loss']:.4f}", flush=True)
    print(f"Done. Best validation F1: {best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
