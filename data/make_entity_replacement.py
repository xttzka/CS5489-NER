"""Build 500 paired CoNLL-2003 test sentences with fixed-seed name replacement.

Match gold type, per-token casing and spaCy POS. Restrict MISC to selected
nationality and regional terms with matching grammatical role. Select one
entity per sentence and prefer donors used fewer times and in similar contexts.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

from data.make_entity_index import DATA_DIR, name_key
from eval.score_ner import bio_to_spans


OUTPUT_DIR = DATA_DIR / "derived" / "entity_replacement" / "conll2003"
TRAIN = DATA_DIR / "processed" / "conll2003" / "train.jsonl"
VERSION = "v2"
SEED = 42
QUOTAS = {"PER": 150, "MISC": 50, "LOC": 150, "ORG": 150}
ORIGINAL_CAP = 3
DONOR_CAP = 3
MISC_DONOR_CAP = 8
MISC_TERMS = (
    "african albanian algerian american asian australian austrian basque bahraini belarusian belgian "
    "brazilian british bulgarian burmese cambodian canadian chilean chinese croatian czech dutch egyptian "
    "english european french german greek hungarian indian indonesian irish israeli italian japanese korean "
    "kuwaiti lebanese liberian mauritian mexican moroccan norwegian omani palestinian polish portuguese "
    "russian rwandan scottish singaporean slovak spanish swiss syrian tasmanian thai turkish ugandan "
    "uruguayan uzbek welsh yugoslav zairean"
).split()
MISC_PLURALS = (
    "africans algerians americans brazilians britons canadians czechs europeans ghanaians hondurans "
    "hungarians indonesians israelis italians koreans norwegians palestinians syrians zaireans"
).split()
MISC_SINGULARS = "argentine croat englishman estonian frenchman queenslander swede welshman".split()
MISC_ROLES = {
    **{name: "adjective" for name in MISC_TERMS},
    **{name: "plural" for name in MISC_PLURALS},
    **{name: "singular" for name in MISC_SINGULARS},
    "south african": "adjective", "south korean": "adjective", "east timorese": "adjective",
    "central american": "adjective", "south american": "adjective", "southern european": "adjective",
}
FIXED_TERMS = {"currency", "dollar", "dollars", "franc", "francs", "guilder", "guilders", "thermal", "won", "yen"}


class Mention(NamedTuple):
    row: dict
    start: int
    end: int
    kind: str
    tokens: list[str]
    name: tuple[str, ...]
    signature: tuple
    context: frozenset


def read_rows(path):
    with path.open(encoding="utf-8-sig") as source:
        yield from map(json.loads, source)


def letter_case(token):
    return token.isupper(), token.istitle(), token.islower()


def occurrences(sequence, name):
    return sum(sequence[i:i + len(name)] == name for i in range(len(sequence) - len(name) + 1))


def check_unseen(names):
    lengths = sorted({len(name) for name in names})
    for row in read_rows(TRAIN):
        sentence = name_key(row["tokens"])
        for length in lengths:
            for start in range(len(sentence) - length + 1):
                if sentence[start:start + length] in names:
                    raise ValueError(f"replacement name occurs in training: {sentence[start:start + length]}")


def test_mentions(rows):
    for row in rows:
        tokens, tags, pos = row["tokens"], row["tags"], row["pos_tags"]
        if len(tokens) != len(tags) or len(tokens) != len(pos):
            raise ValueError(f"{row['sample_id']}: tokens and tags differ in length")
        for start, end, kind in sorted(bio_to_spans(tags)):
            name_tokens = tokens[start:end]
            if not all(token.isalpha() for token in name_tokens):
                continue
            name = name_key(name_tokens)
            pattern = tuple(pos[start:end])
            role = MISC_ROLES.get(" ".join(name)) if kind == "MISC" else None
            if kind == "MISC" and (
                role is None
                or (role == "adjective" and not any(tag.startswith("JJ") for tag in pattern))
                or (role == "plural" and pattern[-1] not in ("NNS", "NNPS"))
                or (role == "singular" and pattern[-1] not in ("NN", "NNP"))
                or (end < len(tokens) and tokens[end].casefold() in FIXED_TERMS)
            ):
                continue
            signature = (kind, tuple(map(letter_case, name_tokens)), pattern, role)
            context = frozenset(name_key(tokens[max(0, start - 4):start] + tokens[end:end + 4]))
            yield Mention(row, start, end, kind, name_tokens, name, signature, context)


def training_references(lengths):
    gold = set()
    text = set()
    for row in read_rows(TRAIN):
        words = name_key(row["tokens"])
        gold.update((kind, words[start:end]) for start, end, kind in bio_to_spans(row["tags"]))
        for length in lengths:
            text.update(words[i:i + length] for i in range(len(words) - length + 1))
    return gold, text


def choose_pairs(mentions, seen, training_text):
    candidates, donors = defaultdict(list), defaultdict(list)
    name_types = defaultdict(set)
    for mention in mentions:
        name_types[mention.name].add(mention.kind)
    for mention in mentions:
        if (mention.kind, mention.name) in seen and mention.row["dataset"] == "conll2003":
            candidates[mention.kind].append(mention)
        if mention.name not in training_text and name_types[mention.name] == {mention.kind}:
            donors[mention.signature].append(mention)

    used_rows, original_use, donor_use = set(), Counter(), Counter()
    rng = random.Random(SEED)
    pairs = []
    for kind, quota in QUOTAS.items():
        rng.shuffle(candidates[kind])
        count = 0
        donor_cap = MISC_DONOR_CAP if kind == "MISC" else DONOR_CAP
        for cap in range(1, ORIGINAL_CAP + 1):
            for original in candidates[kind]:
                if count == quota:
                    break
                sample_id = original.row["sample_id"]
                if sample_id in used_rows or original_use[original.name] >= cap:
                    continue
                sentence = name_key(original.row["tokens"])
                if occurrences(sentence, original.name) != 1:
                    continue
                options = [
                    donor for donor in donors[original.signature]
                    if donor_use[donor.name] < donor_cap
                    and donor.row["doc_id"] != original.row["doc_id"]
                    and not occurrences(sentence, donor.name)
                ]
                if not options:
                    continue
                # Prefer fewer repeats, then more matching words near the entity.
                donor = min(options, key=lambda item: (
                    donor_use[item.name], -len(original.context & item.context),
                ))
                pairs.append((original, donor))
                used_rows.add(sample_id)
                original_use[original.name] += 1
                donor_use[donor.name] += 1
                count += 1
        if count != quota:
            raise ValueError(f"{kind}: only {count}/{quota} pairs meet the v2 selection rules")
    check_unseen(donor_use.keys())
    return sorted(pairs, key=lambda pair: int(pair[0].row["sample_id"].rsplit(":", 1)[1]))


def build_rows(pairs):
    for original, donor in pairs:
        before = original.row
        start, end = original.start, original.end
        tokens, tags = before["tokens"], before["tags"]
        after = {**before, "tokens": tokens[:start] + donor.tokens + tokens[end:]}
        if len(after["tokens"]) != len(tags) or (start, end, original.kind) not in bio_to_spans(tags):
            raise ValueError(f"{before['sample_id']}: replacement broke BIO alignment")
        change = {
            "sample_id": before["sample_id"], "start": start, "end": end, "entity_type": original.kind,
            "original_text": " ".join(original.tokens), "replacement_text": " ".join(donor.tokens),
            "original_tokens": original.tokens, "replacement_tokens": donor.tokens,
            "bio_tags": tags[start:end], "donor_sample_id": donor.row["sample_id"],
            "donor_start": donor.start, "donor_end": donor.end,
        }
        yield before, after, change


def write_pairs(pairs):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filenames = (f"original_{VERSION}.jsonl", f"replaced_{VERSION}.jsonl", f"replacements_{VERSION}.jsonl")
    with TemporaryDirectory(dir=OUTPUT_DIR) as staging:
        pending = []
        for filename, records in zip(filenames, zip(*pairs), strict=True):
            staged, output = Path(staging) / filename, OUTPUT_DIR / filename
            with staged.open("w", encoding="utf-8", newline="\n") as target:
                for record in records:
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
            if output.exists() and output.read_bytes() != staged.read_bytes():
                raise ValueError(f"{output}: frozen file differs; use a new version to change it")
            pending.append((staged, output))
        for staged, output in pending:
            if output.exists():
                print(f"Verified {output}")
            else:
                staged.replace(output)
                print(f"Wrote {output}")
    counts = Counter(change["entity_type"] for _, _, change in pairs)
    print(f"{len(pairs)} sentence pairs: " + ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items())))


def main():
    rows = [row for dataset in ("conll2003", "conllpp")
            for row in read_rows(DATA_DIR / "processed_pos" / dataset / "test.jsonl")]
    mentions = list(test_mentions(rows))
    seen, training_text = training_references(sorted({len(item.tokens) for item in mentions}))
    pairs = choose_pairs(mentions, seen, training_text)
    write_pairs(list(build_rows(pairs)))
    for kind in QUOTAS:
        names = Counter(donor.name for original, donor in pairs if original.kind == kind)
        print(f"{kind}: {len(names)} different donors; most-used donor {max(names.values())} times")


if __name__ == "__main__":
    main()
