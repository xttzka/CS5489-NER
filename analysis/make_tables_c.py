"""Generate member C's tables: zero-shot rows of T3, the LoRA comparison of T4 and the robustness results."""

import csv
import json
from statistics import mean, pstdev

from analysis.data_exploration import ROOT
from analysis.make_tables import OUTPUT, table


RESULTS = ROOT / "results" / "c"
DATASETS = {
    "conll2003": "CoNLL-03", "conllpp": "CoNLL++", "wikigold": "WikiGold",
    "btc": "BTC", "wnut17": "WNUT-17", "mit_restaurant": "MIT Rest.",
}
BEST = "qwen7b_k10_similar_plain"


def f1(name):
    report = json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))
    return f"{100 * report['scores']['f1']:.1f}"


def llm(method, dataset, scope="test_300", split="test"):
    return f1(f"{method}_{scope}_{dataset}_{split}_seed0")


def zero_shot_table():
    groups = json.loads((ROOT / "results" / "m0_m2_baselines.json").read_text(encoding="utf-8"))["groups"]
    sources = {dataset: dataset if dataset in groups else "transfer" for dataset in DATASETS}
    crf = (f"{100 * groups[sources[dataset]]['targets'][dataset]['test_300']['models']['m2']['f1']:.1f}"
           for dataset in DATASETS)
    rows = [
        ("CRF", "Full training split", *crf),
        ("GLiNER", "None", *(f1(f"gliner_{dataset}_test_test_300") for dataset in DATASETS)),
        ("Qwen2.5-1.5B", "None", *(llm("qwen1.5b_k0_plain", dataset) for dataset in DATASETS)),
        ("Qwen2.5-7B", "None", *(llm("qwen7b_k0_plain", dataset) for dataset in DATASETS)),
        ("Qwen2.5-7B", "10 retrieved examples", *(llm(BEST, dataset) for dataset in DATASETS)),
    ]
    return table("ll" + "r" * len(DATASETS), ("Method", "Labelled sentences", *DATASETS.values()), rows,
                 "T3, member C. Entity F1 (%) on the fixed 300-sentence test subsets, native labels. "
                 "Few-shot retrieves 10 examples per sentence from the full training split of the label scheme. "
                 "Load booktabs.")


def pool_table():
    """F1 of the 7B model as the labelled pool that examples are retrieved from grows."""
    sizes = (("50", (0, 1, 2), "50"), ("100", (0, 1, 2), "100"), ("0.01", (0,), "1\\%"), ("0.1", (0,), "10\\%"))
    rows = []
    for dataset in ("conll2003", "wnut17", "mit_restaurant"):
        cells = [llm("qwen7b_k0_plain", dataset)]
        for size, seeds, _ in sizes:
            values = []
            for seed in seeds:
                path = RESULTS / f"{BEST}_pool{size}_test_300_{dataset}_test_seed{seed}.json"
                values.append(100 * json.loads(path.read_text(encoding="utf-8"))["scores"]["f1"])
            cells.append(f"{mean(values):.1f}" + (f" $\\pm$ {pstdev(values):.1f}" if len(values) > 1 else ""))
        rows.append((DATASETS[dataset], *cells, llm(BEST, dataset)))
    return table("lrrrrrr", ("Dataset", "0", *(label for _, _, label in sizes), "100\\%"), rows,
                 "Member C. Entity F1 (%) of Qwen2.5-7B on the fixed 300-sentence test subsets by the number of "
                 "labelled training sentences available as examples. Column 0 uses no examples. The other columns "
                 "retrieve 10 examples per sentence from the pool. 50 and 100 show mean and SD over 3 pools. "
                 "Load booktabs.")


def team_f1(**keys):
    """Read one F1 of member B's result table."""
    with (ROOT / "results" / "B.csv").open(encoding="utf-8-sig") as source:
        rows = [row for row in csv.DictReader(source) if row["metric"] == "f1" and row["entity_type"] == "all"
                and row["entity_group"] == "all" and all(row[key] == value for key, value in keys.items())]
    if len(rows) != 1:
        raise ValueError(f"expected one F1 row for {keys}, found {len(rows)}")
    return f"{100 * float(rows[0]['value']):.1f}"


def distill_table():
    """BERT trained on 2,000 MIT Restaurant queries with gold labels and with the LLM's labels."""
    labels = json.loads((RESULTS / "qwen7b_k0_plain_train_2000_v1_mit_restaurant_validation_seed0.json")
                        .read_text(encoding="utf-8"))["scores"]
    keys = dict(experiment="distill", dataset="mit_restaurant", scope="test_300")
    rows = [
        ("7B LLM, no examples", "--", llm("qwen7b_k0_plain", "mit_restaurant")),
        ("BERT on the LLM's labels", f"{100 * labels['f1']:.1f}", team_f1(condition="pseudo", **keys)),
        ("BERT on gold labels", "100.0", team_f1(condition="gold", **keys)),
    ]
    return table("lrr", ("Extractor", "Training label F1", "Test F1"), rows,
                 "T4, members B and C. Entity F1 (%) on 300 MIT Restaurant test queries. Both BERT taggers train "
                 "on the same 2,000 queries. Load booktabs.")


def lora_table():
    targets = (("conll2003", ""), ("conllpp", ""), ("wikigold", ""), ("btc", "_as_conll2003"), ("wnut17", "_as_conll2003"))
    rows = [
        ("Qwen2.5-1.5B prompt, no examples", *(llm(f"qwen1.5b_k0_plain{suffix}", dataset) for dataset, suffix in targets)),
        ("Qwen2.5-1.5B LoRA, 10\\% CoNLL-03", *(llm(f"lora1.5b_k0_plain{suffix}", dataset) for dataset, suffix in targets)),
        ("BERT, 10\\% CoNLL-03", *(team_f1(experiment="lora_match", dataset=dataset, scope="test_300")
                                    for dataset, _ in targets)),
    ]
    return table("lrrrrr", ("Extractor", *(DATASETS[dataset] for dataset, _ in targets)), rows,
                 "T4, member C. Entity F1 (%) on the fixed 300-sentence test subsets. BTC and WNUT-17 are scored "
                 "on the shared PER, LOC and ORG types with CoNLL-03 output labels. Load booktabs.")


def robustness_table():
    scopes = (
        ("full", "test", "Original"), ("lowercase_v1", "validation", "Lowercase"),
        ("spelling_noise_v1", "validation", "5\\% spelling"), ("no_punctuation_v1", "validation", "No punct."),
        ("replacement_original_v2", "validation", "Before"), ("replacement_replaced_v2", "validation", "After"),
    )
    rows = [
        ("GLiNER", *(f1(f"gliner_conll2003_test_{scope}") for scope, _, _ in scopes)),
        ("Qwen2.5-7B, no examples", *(llm("qwen7b_k0_plain", "conll2003", scope, split) for scope, split, _ in scopes)),
        ("Qwen2.5-7B, 10 retrieved", *(llm(BEST, "conll2003", scope, split) for scope, split, _ in scopes)),
    ]
    return table("lrrrrrr", ("Method", *(label for _, _, label in scopes)), rows,
                 "Member C. Entity F1 (%) on the full CoNLL-03 test set, its three perturbed versions, and the 500 "
                 "sentence pairs before and after entity replacement. Load booktabs.")


def wording_table():
    """Mean and SD of F1 over three equivalent phrasings of the instructions."""
    rows = []
    for label, method in (("No examples", "qwen7b_k0_plain"), ("10 retrieved examples", BEST)):
        cells = []
        for dataset in ("conll2003", "wnut17", "mit_restaurant"):
            values = []
            for suffix in ("", "_w1", "_w2"):
                path = RESULTS / f"{method}{suffix}_test_300_{dataset}_test_seed0.json"
                values.append(100 * json.loads(path.read_text(encoding="utf-8"))["scores"]["f1"])
            cells.append(f"{mean(values):.1f} $\\pm$ {pstdev(values):.1f}")
        rows.append((label, *cells))
    return table("lrrr", ("Qwen2.5-7B", DATASETS["conll2003"], DATASETS["wnut17"], DATASETS["mit_restaurant"]), rows,
                 "Member C. Entity F1 (%) on the fixed 300-sentence test subsets, mean and SD over three "
                 "equivalent phrasings of the instructions. Load booktabs.")


def main():
    tables = {"t3_c": zero_shot_table(), "t4_c": lora_table(), "robustness_c": robustness_table(),
              "pool_c": pool_table(), "wording_c": wording_table(), "distill_c": distill_table()}
    for key, content in tables.items():
        (OUTPUT / f"{key}.tex").write_text(content, encoding="utf-8", newline="\n")
        print(OUTPUT / f"{key}.tex")


if __name__ == "__main__":
    main()
