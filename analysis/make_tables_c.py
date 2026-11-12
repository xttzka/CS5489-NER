"""Generate member C's tables: zero-shot rows of T3, the LoRA comparison of T4 and the robustness results."""

import json

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
    rows = [
        ("GLiNER", "None", *(f1(f"gliner_{dataset}_test_test_300") for dataset in DATASETS)),
        ("Qwen2.5-1.5B", "None", *(llm("qwen1.5b_k0_plain", dataset) for dataset in DATASETS)),
        ("Qwen2.5-7B", "None", *(llm("qwen7b_k0_plain", dataset) for dataset in DATASETS)),
        ("Qwen2.5-7B", "10 retrieved", *(llm(BEST, dataset) for dataset in DATASETS)),
    ]
    return table("ll" + "r" * len(DATASETS), ("Method", "Examples", *DATASETS.values()), rows,
                 "T3, member C. Entity F1 (%) on the fixed 300-sentence test subsets, native labels. "
                 "Retrieved examples come from the training split of the label scheme. Load booktabs.")


def lora_table():
    targets = (("conll2003", ""), ("conllpp", ""), ("wikigold", ""), ("btc", "_as_conll2003"), ("wnut17", "_as_conll2003"))
    training = json.loads((RESULTS / "lora_train_conll2003_n0.1_seed0.json").read_text(encoding="utf-8"))
    rows = [
        ("Prompt, no examples", "0", *(llm(f"qwen1.5b_k0_plain{suffix}", dataset) for dataset, suffix in targets)),
        ("LoRA, 10\\% CoNLL-03", f"{training['timing']['trainable_parameter_count'] / 1e6:.1f}M",
         *(llm(f"lora1.5b_k0_plain{suffix}", dataset) for dataset, suffix in targets)),
    ]
    return table("llrrrrr", ("Qwen2.5-1.5B", "Trained", *(DATASETS[dataset] for dataset, _ in targets)), rows,
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


def main():
    tables = {"t3_c": zero_shot_table(), "t4_c": lora_table(), "robustness_c": robustness_table()}
    for key, content in tables.items():
        (OUTPUT / f"{key}.tex").write_text(content, encoding="utf-8", newline="\n")
        print(OUTPUT / f"{key}.tex")


if __name__ == "__main__":
    main()
