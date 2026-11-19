"""GloVe + character CNN + BiLSTM-CRF for NER."""

import argparse
import json
import random
import time
from hashlib import file_digest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torchcrf import CRF

from eval.score_ner import score_ner
from models.llm.common import TYPE_DESCRIPTIONS, load_rows, sample_train
from models.neural.bert_baseline import fix_iob2


GLOVE_PATH = Path("data/external/glove/glove.6B.100d.txt")
GLOVE_SHA256 = (
    "be4367dd257eb945217234f16307c5c74236b648a222cc0b4ffd0dda6a3350b6"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a GloVe + character CNN + BiLSTM-CRF NER model."
    )

    parser.add_argument(
        "--dataset",
        default="conll2003",
        choices=["conll2003", "wnut17", "mit_restaurant"],
    )
    parser.add_argument(
        "--train-size",
        default="200",
        help='Sentence count such as "200", or proportion such as "1.0".',
    )
    parser.add_argument(
        "--dev-size",
        type=int,
        default=100,
        help="Validation sentence count; 0 means the full validation set.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="output/bilstm_conll2003_smoke",
    )

    return parser.parse_args()


def load_glove(path):
    """Load the complete GloVe vocabulary and its frozen 100d vectors."""

    if not path.is_file():
        raise FileNotFoundError(f"GloVe file not found: {path}")

    print("Checking GloVe SHA-256...", flush=True)

    with path.open("rb") as file:
        actual_sha256 = file_digest(file, "sha256").hexdigest()

    if actual_sha256 != GLOVE_SHA256:
        raise ValueError(
            f"GloVe SHA-256 mismatch: {actual_sha256}"
        )

    print("Counting GloVe entries...", flush=True)

    with path.open(encoding="utf-8") as file:
        entry_count = sum(1 for _ in file)

    # Index 0 is reserved for padding.
    vectors = np.zeros((entry_count + 1, 100), dtype=np.float32)
    word_to_id = {"<pad>": 0}

    print(f"Loading {entry_count:,} GloVe vectors...", flush=True)

    with path.open(encoding="utf-8") as file:
        for index, line in enumerate(file, start=1):
            parts = line.rstrip().split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(f"Invalid GloVe line: {index}")

            word, values = parts

            if word in word_to_id:
                raise ValueError(f"Duplicate or reserved GloVe word: {word}")

            vector = np.fromstring(values, sep=" ", dtype=np.float32)

            if vector.shape != (100,):
                raise ValueError(
                    f"GloVe line {index} has {len(vector)} values, expected 100"
                )

            word_to_id[word] = index
            vectors[index] = vector

    if "<unk>" not in word_to_id:
        raise ValueError("The GloVe file does not contain the expected <unk>.")

    return word_to_id, torch.from_numpy(vectors), actual_sha256


def make_character_vocab(rows):
    """Build the character vocabulary from training sentences only."""

    characters = sorted(
        {
            character
            for row in rows
            for token in row["tokens"]
            for character in token
        }
    )

    char_to_id = {"<pad>": 0, "<unk>": 1}

    for character in characters:
        char_to_id[character] = len(char_to_id)

    return char_to_id


def encode_batch(rows, word_to_id, char_to_id, label_to_id, device):
    batch_size = len(rows)
    max_words = max(len(row["tokens"]) for row in rows)

    max_chars = max(
        len(token)
        for row in rows
        for token in row["tokens"]
    )

    word_ids = torch.zeros(
        (batch_size, max_words),
        dtype=torch.long,
    )
    char_ids = torch.zeros(
        (batch_size, max_words, max_chars),
        dtype=torch.long,
    )
    tag_ids = torch.zeros(
        (batch_size, max_words),
        dtype=torch.long,
    )
    mask = torch.zeros(
        (batch_size, max_words),
        dtype=torch.bool,
    )

    lengths = torch.tensor(
        [len(row["tokens"]) for row in rows],
        dtype=torch.long,
    )

    unknown_word_id = word_to_id["<unk>"]
    unknown_char_id = char_to_id["<unk>"]

    for batch_index, row in enumerate(rows):
        tokens = row["tokens"]
        tags = row["tags"]

        if len(tokens) != len(tags) or not tokens:
            raise ValueError(
                f'{row["sample_id"]}: invalid token/tag alignment'
            )

        for word_index, (token, tag) in enumerate(zip(tokens, tags)):
            word_ids[batch_index, word_index] = word_to_id.get(
                token.lower(),
                unknown_word_id,
            )
            tag_ids[batch_index, word_index] = label_to_id[tag]
            mask[batch_index, word_index] = True

            for char_index, character in enumerate(token):
                char_ids[batch_index, word_index, char_index] = char_to_id.get(
                    character,
                    unknown_char_id,
                )

    return (
        word_ids.to(device),
        char_ids.to(device),
        tag_ids.to(device),
        mask.to(device),
        lengths,
    )


class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        glove_vectors,
        num_characters,
        num_labels,
        hidden_size,
        dropout,
    ):
        super().__init__()

        # Frozen GloVe vectors; padding uses index 0.
        self.word_embedding = nn.Embedding.from_pretrained(
            glove_vectors,
            freeze=True,
            padding_idx=0,
        )

        self.char_embedding = nn.Embedding(
            num_characters,
            30,
            padding_idx=0,
        )

        self.char_cnn = nn.Conv1d(
            in_channels=30,
            out_channels=30,
            kernel_size=3,
            padding=1,
        )

        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=130,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def emissions(self, word_ids, char_ids, lengths):
        batch_size, max_words, max_chars = char_ids.shape

        word_vectors = self.word_embedding(word_ids)

        flat_chars = char_ids.reshape(batch_size * max_words, max_chars)
        valid_chars = flat_chars.ne(0)

        char_vectors = self.char_embedding(flat_chars)
        char_vectors = char_vectors.transpose(1, 2)

        char_features = torch.relu(self.char_cnn(char_vectors))

        # Exclude padding character positions from max pooling.
        char_features = char_features.masked_fill(
            ~valid_chars.unsqueeze(1),
            -1e4,
        )

        char_features = char_features.max(dim=-1).values

        # A fully padded word has no characters; use an all-zero vector.
        has_characters = valid_chars.any(dim=-1, keepdim=True)
        char_features = torch.where(
            has_characters,
            char_features,
            torch.zeros_like(char_features),
        )

        char_features = char_features.reshape(batch_size, max_words, 30)

        inputs = torch.cat([word_vectors, char_features], dim=-1)
        inputs = self.dropout(inputs)

        packed = pack_padded_sequence(
            inputs,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.bilstm(packed)

        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_words,
        )

        return self.classifier(self.dropout(output))


def evaluate(
    model,
    loader,
    word_to_id,
    char_to_id,
    label_to_id,
    id_to_label,
    device,
):
    model.eval()

    gold_by_id = {}
    predictions_by_id = {}

    with torch.no_grad():
        for rows in loader:
            word_ids, char_ids, _, mask, lengths = encode_batch(
                rows,
                word_to_id,
                char_to_id,
                label_to_id,
                device,
            )

            emissions = model.emissions(word_ids, char_ids, lengths)
            decoded = model.crf.decode(emissions, mask=mask)

            for row, sequence in zip(rows, decoded):
                tags = [id_to_label[index] for index in sequence]

                if len(tags) != len(row["tokens"]):
                    raise ValueError(
                        f'{row["sample_id"]}: decoded length mismatch'
                    )

                gold_by_id[row["sample_id"]] = row["tags"]
                predictions_by_id[row["sample_id"]] = fix_iob2(tags)

    scores = score_ner(gold_by_id, predictions_by_id)
    return scores, predictions_by_id


def main():
    args = parse_args()

    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.dev_size < 0
        or args.hidden_size < 1
    ):
        raise ValueError("Invalid training arguments.")

    output_dir = Path(args.output_dir)

    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; choose a new output directory."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    train_rows = sample_train(args.dataset, args.train_size, args.seed)
    dev_rows = load_rows(args.dataset, "validation")

    if args.dev_size:
        dev_rows = dev_rows[:args.dev_size]

    if not train_rows or not dev_rows:
        raise ValueError("Training and validation sets must not be empty.")

    entity_types = sorted(TYPE_DESCRIPTIONS[args.dataset])

    labels = ["O"]

    for entity_type in entity_types:
        labels.extend([f"B-{entity_type}", f"I-{entity_type}"])

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }
    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    char_to_id = make_character_vocab(train_rows)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Training sentences: {len(train_rows)}")
    print(f"Validation sentences: {len(dev_rows)}")
    print(f"Labels: {labels}")
    print(f"Character vocabulary size: {len(char_to_id)}")

    word_to_id, glove_vectors, glove_sha256 = load_glove(GLOVE_PATH)

    print(f"GloVe vocabulary size: {len(word_to_id) - 1:,}")
    print("GloVe vectors are frozen during training.")

    model = BiLSTMCRF(
        glove_vectors=glove_vectors,
        num_characters=len(char_to_id),
        num_labels=len(labels),
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    train_loader = DataLoader(
        train_rows,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=list,
    )
    dev_loader = DataLoader(
        dev_rows,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=list,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    output_dir.mkdir(parents=True)

    config = vars(args).copy()
    config["device"] = str(device)
    config["glove_path"] = str(GLOVE_PATH)
    config["glove_sha256"] = glove_sha256
    config["glove_frozen"] = True
    config["labels"] = labels

    with (output_dir / "run_config.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        epoch_loss = 0.0

        for batch_number, rows in enumerate(train_loader, start=1):
            word_ids, char_ids, tag_ids, mask, lengths = encode_batch(
                rows,
                word_to_id,
                char_to_id,
                label_to_id,
                device,
            )

            optimizer.zero_grad()

            emissions = model.emissions(word_ids, char_ids, lengths)

            loss = -model.crf(
                emissions,
                tag_ids,
                mask=mask,
                reduction="token_mean",
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
                foreach=False,
            )

            optimizer.step()
            epoch_loss += loss.item()

            if batch_number == 1 or batch_number % 10 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"batch {batch_number}/{len(train_loader)} | "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )

        scores, predictions = evaluate(
            model,
            dev_loader,
            word_to_id,
            char_to_id,
            label_to_id,
            id_to_label,
            device,
        )

        print(
            f"Epoch {epoch} finished in {time.time() - started:.1f}s | "
            f"mean loss: {epoch_loss / len(train_loader):.4f}"
        )
        print("Validation scores:")
        print(json.dumps(scores, indent=2))

        if scores["f1"] > best_f1:
            best_f1 = scores["f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "word_to_id": word_to_id,
                    "char_to_id": char_to_id,
                    "label_to_id": label_to_id,
                    "config": config,
                    "best_epoch": epoch,
                },
                output_dir / "best_model.pt",
            )

            with (output_dir / "validation_scores.json").open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(scores, file, indent=2)

            with (output_dir / "validation_predictions.jsonl").open(
                "w", encoding="utf-8"
            ) as file:
                for sample_id, tags in predictions.items():
                    record = {"sample_id": sample_id, "tags": tags}
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Best model saved to: {output_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
