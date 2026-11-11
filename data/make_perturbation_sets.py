"""Freeze three paired CoNLL-2003 test variants and their change log."""

import json
import random
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from string import ascii_letters, ascii_lowercase, ascii_uppercase
from tempfile import TemporaryDirectory
from unicodedata import category

from data.make_entity_index import DATA_DIR
from eval.score_ner import bio_to_spans


SOURCE = DATA_DIR / "processed_pos" / "conll2003" / "test.jsonl"
OUTPUT = DATA_DIR / "derived" / "perturbations" / "conll2003"
SEED = 42
NOISE_RATE = 0.05
ALIGNED_FIELDS = ("tokens", "tags", "pos_tags", "chunk_tags")


def make_noise_plan(rows):
    positions = [
        (row_index, token_index, character_index)
        for row_index, row in enumerate(rows)
        for token_index, token in enumerate(row["tokens"])
        for character_index, character in enumerate(token)
        if character in ascii_letters
    ]
    rng = random.Random(SEED)
    selected = sorted(rng.sample(positions, round(len(positions) * NOISE_RATE)))
    edits = defaultdict(list)
    for row_index, token_index, character_index in selected:
        before = rows[row_index]["tokens"][token_index][character_index]
        alphabet = ascii_uppercase if before.isupper() else ascii_lowercase
        after = rng.choice(alphabet.replace(before, ""))
        edits[row_index].append({
            "token_index": token_index, "character_index": character_index,
            "before": before, "after": after,
        })
    return edits, len(positions), len(selected)


def lowercase_row(row):
    tokens = [token.lower() for token in row["tokens"]]
    edits = [
        {"token_index": index, "before": before, "after": after}
        for index, (before, after) in enumerate(zip(row["tokens"], tokens, strict=True))
        if before != after
    ]
    return {**row, "tokens": tokens}, edits


def noisy_row(row, edits):
    characters = [list(token) for token in row["tokens"]]
    for edit in edits:
        characters[edit["token_index"]][edit["character_index"]] = edit["after"]
    return {**row, "tokens": ["".join(token) for token in characters]}


def is_punctuation(token):
    return bool(token) and all(category(character).startswith("P") for character in token)


def project_spans(spans, original_to_new):
    projected = set()
    for start, end, kind in spans:
        surviving = [original_to_new[i] for i in range(start, end) if i in original_to_new]
        if surviving:
            projected.add((surviving[0], surviving[-1] + 1, kind))
    return projected


def remove_punctuation(row):
    tokens, tags = row["tokens"], row["tags"]
    removed = {i for i, (token, tag) in enumerate(zip(tokens, tags, strict=True))
               if tag == "O" and is_punctuation(token)}
    retain_empty = len(removed) == len(tokens)
    if retain_empty:
        removed.clear()
    kept = [i for i in range(len(tokens)) if i not in removed]
    original_to_new = {old: new for new, old in enumerate(kept)}
    after = {**row, **{key: [row[key][i] for i in kept] for key in ALIGNED_FIELDS}}

    # If a deleted token began a chunk, the first surviving token now begins it.
    chunk_spans = project_spans(bio_to_spans(row["chunk_tags"]), original_to_new)
    for start, _, kind in chunk_spans:
        after["chunk_tags"][start] = f"B-{kind}"
    if bio_to_spans(after["chunk_tags"]) != chunk_spans:
        raise ValueError(f"{row['sample_id']}: projected chunk tags differ")
    if bio_to_spans(after["tags"]) != project_spans(bio_to_spans(tags), original_to_new):
        raise ValueError(f"{row['sample_id']}: projected entity tags differ")

    return after, {
        "removed": [{"token_index": i, "token": tokens[i], "tag": tags[i]} for i in sorted(removed)],
        "new_to_original_token_index": kept,
        "retained_to_avoid_empty_sentence": retain_empty,
        "chunk_prefix_changes": [
            {"token_index": old, "before": row["chunk_tags"][old], "after": after["chunk_tags"][new]}
            for new, old in enumerate(kept)
            if row["chunk_tags"][old] != after["chunk_tags"][new]
        ],
    }


def validate_row(row):
    size = len(row["tokens"])
    if not size or any(len(row[key]) != size for key in ALIGNED_FIELDS):
        raise ValueError(f"{row['sample_id']}: tokens and aligned tags differ in length")
    bio_to_spans(row["tags"])


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    source_bytes = SOURCE.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode("utf-8-sig").splitlines()]
    ids = [row["sample_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or any(
        row["dataset"] != "conll2003" or row["split"] != "test" for row in rows
    ):
        raise ValueError("expected unique CoNLL-2003 test records")
    for row in rows:
        validate_row(row)
    noise, eligible, sampled = make_noise_plan(rows)
    lowercase, spelling, punctuation, changes = [], [], [], []
    counts = Counter()
    for index, row in enumerate(rows):
        low, lower_edits = lowercase_row(row)
        noise_edits = noise[index]
        noisy = noisy_row(row, noise_edits)
        punct, removed = remove_punctuation(row)
        for variant in (low, noisy, punct):
            validate_row(variant)
        lowercase.append(low)
        spelling.append(noisy)
        punctuation.append(punct)
        changes.append({
            "sample_id": row["sample_id"], "lowercase": lower_edits,
            "spelling_noise": noise_edits, "no_punctuation": removed,
        })
        counts["lowercase_tokens"] += len(lower_edits)
        counts["lowercase_sentences"] += bool(lower_edits)
        counts["spelling_sentences"] += bool(noise_edits)
        counts["spelling_letters"] += len(noise_edits)
        counts["punctuation_tokens_removed"] += len(removed["removed"])
        counts["punctuation_sentences"] += bool(removed["removed"])
        counts["punctuation_only_sentences_retained"] += removed["retained_to_avoid_empty_sentence"]
        counts["chunk_prefixes_changed"] += len(removed["chunk_prefix_changes"])
    if counts["spelling_letters"] != sampled:
        raise ValueError("number of recorded spelling edits differs from the sample")

    files = {
        "lowercase_v1.jsonl": lowercase, "spelling_noise_v1.jsonl": spelling,
        "no_punctuation_v1.jsonl": punctuation, "changes_v1.jsonl": changes,
    }
    manifest = {
        "source": SOURCE.relative_to(DATA_DIR.parent).as_posix(),
        "source_sha256": sha256(source_bytes).hexdigest(),
        "sentences_per_variant": len(rows),
        "source_tokens": sum(len(row["tokens"]) for row in rows),
        "spelling_noise": {"seed": SEED, "rate": NOISE_RATE, "eligible_ascii_letters": eligible,
                           "rule": "Sample positions corpus-wide without replacement; use a different letter of the same case."},
        "punctuation_rule": "Remove standalone Unicode punctuation tagged O; retain the original row if it would become empty.",
        "alignment": "Zero-based token and character offsets. POS is frozen from the original text. "
                     "Deletion projects the existing chunk tags to surviving positions.",
        "counts": dict(counts),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT) as staging:
        work_dir = Path(staging)
        for filename, records in files.items():
            write_jsonl(work_dir / filename, records)
        manifest["sha256"] = {filename: sha256((work_dir / filename).read_bytes()).hexdigest() for filename in files}
        metadata = "manifest_v1.json"
        (work_dir / metadata).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for filename in (*files, metadata):
            target = OUTPUT / filename
            if target.exists() and target.read_bytes() != (work_dir / filename).read_bytes():
                raise ValueError(f"{target}: frozen file differs; use a new version to change it")
        for filename in (*files, metadata):
            target = OUTPUT / filename
            if not target.exists():
                (work_dir / filename).replace(target)
    print(f"{len(rows):,} sentences in each variant. Output: {OUTPUT}")
    print(f"Lowercased {counts['lowercase_tokens']:,} tokens; changed {sampled:,}/{eligible:,} letters; "
          f"removed {counts['punctuation_tokens_removed']:,} punctuation tokens.")


if __name__ == "__main__":
    main()
