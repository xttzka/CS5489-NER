"""Freeze a 300-sentence evaluation subset of each processed test split."""

import json
import random
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DATASETS = ("conll2003", "conllpp", "wnut17", "mit_restaurant", "wikigold", "btc")
SUBSET_SIZE = 300
SEED = 42


def select_test_ids(dataset, path):
    sample_ids = set()
    prefix = f"{dataset}:test:"
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{location}: expected a JSON object")
            if record.get("dataset") != dataset or record.get("split") != "test":
                raise ValueError(f"{location}: expected the {dataset} test split")

            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id.startswith(prefix):
                raise ValueError(f"{location}: invalid test sample ID {sample_id!r}")
            index = sample_id[len(prefix):]
            if not index.isascii() or not index.isdigit():
                raise ValueError(f"{location}: invalid test sample ID {sample_id!r}")
            if sample_id in sample_ids:
                raise ValueError(f"{location}: duplicate sample ID {sample_id!r}")
            sample_ids.add(sample_id)

    if len(sample_ids) < SUBSET_SIZE:
        raise ValueError(f"{path}: need at least {SUBSET_SIZE} IDs, got {len(sample_ids)}")
    return sorted(random.Random(SEED).sample(sorted(sample_ids), SUBSET_SIZE))


def main():
    new_lists = {}
    for dataset in DATASETS:
        source = DATA_DIR / "processed" / dataset / "test.jsonl"
        target = DATA_DIR / "subsets" / dataset / f"test_{SUBSET_SIZE}.txt"
        content = ("\n".join(select_test_ids(dataset, source)) + "\n").encode("utf-8")
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f"{target}: existing IDs differ from the fixed selection")
            print(f"Verified {target}")
        else:
            new_lists[target] = content

    for target, content in new_lists.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(content)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
