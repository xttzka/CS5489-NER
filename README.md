# Multi-Domain Named Entity Recognition from CRF to LLMs

CS5489 course project, Group 59. The project builds named entity extractors of four kinds and compares them on six datasets: a dictionary, logistic regression and a CRF, a BiLSTM-CRF, a fine-tuned BERT tagger and a span classifier, and zero-shot and few-shot extraction with GLiNER and Qwen2.5.

All commands run from the repository root with Python 3.11.

## 1. Repository layout

| Path | Content |
|---|---|
| `data/` | Download, conversion and preparation scripts, the label mapping and the fixed 300-sentence ID lists |
| `models/classic/` | Dictionary (M0), logistic regression (M1), CRF (M2) and their analyses |
| `models/neural/` | BiLSTM-CRF (M3), BERT tagger (M4), span classifier and their analyses |
| `models/llm/` | GLiNER, Qwen2.5 prompting, LoRA fine-tuning |
| `eval/` | Entity-level scoring, bootstrap intervals, error typing, voting |
| `analysis/` | Data exploration, entity clustering, figures and tables |
| `scripts/predict.py` | Entity extraction from raw text |
| `results/` | Scores of every experiment. `A.csv`, `B.csv` and `C.csv` hold all of them in one schema |
| `report/` | Figures and LaTeX tables generated from `results/` |

Datasets, model weights and prediction files are not part of the repository. The commands below create them under `data/`, `output/` and `preds/`.

## 2. Environments

Each requirements file is one virtual environment, because they pin different versions of NumPy and PyTorch.

| File | Used for | Hardware |
|---|---|---|
| `requirements.txt` | Data preparation, classic models, scoring, figures and tables | CPU |
| `requirements-pos.txt` | spaCy part-of-speech tags | CPU |
| `requirements-neural.txt` | `models/neural/` | GPU or Apple MPS recommended |
| `requirements-llm.txt` | `models/llm/` | CUDA GPU, 24 GB for the 7B model |

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3.11 -m venv .venv-pos && .venv-pos/bin/pip install -r requirements-pos.txt
```

On Windows the interpreters are `.venv\Scripts\python.exe` and `.venv-pos\Scripts\python.exe`. Below, `python` is the interpreter of the environment named in each section, and `$POS` is the interpreter of the part-of-speech environment.

## 3. Data

### Download and placement

One command downloads the six corpora into `data/raw/`, converts them and builds every derived set.

```bash
python -B -m data.prepare --all --download --pos-python $POS
```

If a download fails, place the files by hand at the paths below and run the command without `--download`.

| Dataset | Train / validation / test sentences | Files under `data/raw/` | Source |
|---|---:|---|---|
| CoNLL-2003 | 14,041 / 3,250 / 3,453 | `conll2003/conll2003.zip` | <https://huggingface.co/datasets/eriktks/conll2003> |
| WNUT-17 | 3,394 / 1,009 / 1,287 | `wnut17/wnut17train.conll`, `emerging.dev.conll`, `emerging.test.annotated` | <https://github.com/leondz/emerging_entities_17> |
| MIT Restaurant | 6,894 / 766 / 1,521 | `mit_restaurant/restauranttrain.bio`, `restauranttest.bio` | <https://sls.csail.mit.edu/downloads/restaurant/> |
| CoNLL++ | test 1,840 | `conllpp/conllpp.txt` | <https://github.com/ShuhengL/acl2023_conllpp> |
| WikiGold | test 1,696 | `wikigold/wikigold.conll.txt` | <https://github.com/juand-r/entity-recognition-datasets/tree/master/data/wikigold> |
| Broad Twitter Corpus | test 2,000 | `btc/f.json` | <https://github.com/GateNLP/broad_twitter_corpus> |

MIT Restaurant has no validation split, so the conversion holds out 766 training queries.

Two experiments need extra resources.

| Resource | Place at | Source |
|---|---|---|
| GloVe 6B, 100 dimensions | `data/external/glove/glove.6B.100d.txt`, extracted from `glove.6B.zip` | <https://huggingface.co/stanfordnlp/glove/blob/main/glove.6B.zip> |
| GeoNames cities and countries | `data/external/gazetteers/cities15000.zip`, `countryInfo.txt` | <https://download.geonames.org/export/dump/> |

GeoNames updates its files daily. The reported gazetteer results use the download of 10 November 2026, and a later download can change them slightly.

### What the preparation writes

| Output | Content | Script |
|---|---|---|
| `data/processed/<dataset>/<split>.jsonl` | One sentence per line with `sample_id`, `tokens` and IOB2 `tags` | `data/prepare_conll2003.py`, `data/prepare_external_ner.py` |
| `data/processed_pos/<dataset>/<split>.jsonl` | The same sentences with spaCy part-of-speech tags | `data/add_pos_tags.py` |
| `data/subsets/<dataset>/test_300.txt` | Fixed 300 test sentences per dataset, seed 42 | `data/make_eval_subsets.py` |
| `data/derived/entity_index/` | Each test entity marked as seen or unseen in CoNLL-2003 training data | `data/make_entity_index.py` |
| `data/derived/entity_replacement/conll2003/` | 500 sentence pairs before and after each entity is replaced by an unseen name | `data/make_entity_replacement.py` |
| `data/derived/perturbations/conll2003/` | The CoNLL-2003 test set in lowercase, with 5% spelling noise, and without punctuation | `data/make_perturbation_sets.py` |

`python -B -m data.prepare --derived --pos-python $POS` rebuilds the derived sets from existing `data/processed/` files.

## 4. Prediction format and scoring

Every extractor writes `preds/<dataset>/<method>/<seed>.jsonl` with one line per sentence.

```json
{"sample_id": "conll2003:test:0", "tags": ["O", "B-LOC", "O"]}
```

An entity counts as correct when its span and its type both match the gold annotation. The examples below score predictions that section 5 writes.

```bash
# Exact-match precision, recall and F1. Add --subset-ids for the fixed 300 sentences.
python -B -m eval.score_ner --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/m2_baseline/0.jsonl
# Scoring on the shared PER, LOC and ORG types across label schemes
python -B -m eval.score_transfer --gold data/processed/wnut17/test.jsonl --pred preds/wnut17/m2_pos_transfer/0.jsonl --target-dataset wnut17 --prediction-schema conll2003
# Seen and unseen entities
python -B -m eval.score_familiarity --dataset conllpp --pred preds/conllpp/m1_temporal/0.jsonl
# Error types: missed, spurious, boundary, type, boundary and type
python -B -m eval.analyze_ner_errors --gold data/processed/conll2003/test.jsonl --pred preds/conll2003/m2_baseline/0.jsonl --output-dir results/error_analysis/conll2003_m2
```

## 5. Classic models

Environment: `requirements.txt`.

```bash
# Hyperparameter selection: 5-fold cross-validation on the training split
python -B -m models.classic.tune_crf --jobs 3
python -B -m models.classic.m2_native_cv --jobs 3
# M0, M1 and M2 on the three full test sets and the six fixed subsets
python -B -m models.classic.run_baselines --jobs 1
# Cross-validation controls: context window, fold grouping, part-of-speech source
python -B -m models.classic.m2_cv_controls --jobs 3
python -B -m models.classic.m2_pos_cv_check --jobs 5
```

Analyses, in this order because later ones read the predictions of earlier ones:

```bash
python -B -m models.classic.m2_pos_transfer          # CoNLL-2003 model on five test sets
python -B -m eval.score_m2_familiarity               # seen and unseen entities
python -B -m models.classic.run_temporal_baselines   # CoNLL-2003 against CoNLL++
python -B -m eval.analyze_transfer_recall            # recall by familiarity and casing
python -B -m models.classic.m2_transfer_matrix       # three training sources, five targets
python -B -m eval.analyze_m2_transfer_overlap        # entity overlap against the F1 drop
python -B -m models.classic.m2_feature_ablation --jobs 3
python -B -m models.classic.m2_g2_transfer           # word-shape features removed
python -B -m models.classic.m2_pos_validation
python -B -m models.classic.m2_pos_casing
python -B -m models.classic.run_robustness --replacement-version v2
python -B -m models.classic.m2_extra_features        # gazetteer and GloVe cluster features
python -B -m analysis.cluster_entities               # clustering of entity vectors
python -B -m eval.ensemble_ner --gold data/processed/conll2003/test.jsonl --subset-ids data/subsets/conll2003/test_300.txt \
  --pred M0=preds/conll2003/m0_baseline/0.jsonl --pred M1=preds/conll2003/m1_baseline/0.jsonl --pred M2=preds/conll2003/m2_baseline/0.jsonl \
  --group classical=M0,M1,M2 --output results/m0_m2_ensemble.json
python -B -m eval.result_tables export-a             # results/A.csv
```

`tune_crf` and `m2_feature_ablation` reuse the cross-validation scores saved in `results/m2_cv*.json` and `results/m2_feature_ablation.json` when the training data, the code and the library versions of `requirements.txt` are unchanged. Delete these files to run the cross-validation again. Logistic regression scores can differ by about 0.1 F1 between platforms.

## 6. Neural models

Environment: `requirements-neural.txt`. The scripts pick CUDA, then Apple MPS, then the CPU, and they refuse to overwrite an existing output file or directory. `DATASET` is `conll2003`, `wnut17` or `mit_restaurant`, and the reported runs use seeds 0, 1 and 2. Each model has two candidate settings, and the one with the higher validation F1 at seed 0 is used. The commands show the selected one, which is the same on all three datasets. `results/b/hyperparameter_selection.json` holds the validation scores of both.

```bash
# BiLSTM-CRF (M3). The other candidate: lr 0.0005, hidden 96, dropout 0.5
python -B -m models.neural.bilstm_crf --dataset conll2003 --epochs 5 --batch-size 16 \
  --learning-rate 0.001 --hidden-size 128 --dropout 0.3 --seed 0 --output-dir output/bilstm_conll2003_seed0

# BERT tagger (M4). The other candidate: batch 8, lr 3e-5
python -B -m models.neural.bert_extended --task native --dataset conll2003 --train-size 1.0 --dev-size 0 \
  --epochs 3 --batch-size 16 --learning-rate 2e-5 --seed 0 --output-dir output/bert_conll2003_seed0

# Span classifier. The other candidate: batch 2, lr 2e-5
python -B -m models.neural.span_bert --dataset conll2003 --epochs 3 --batch-size 4 --learning-rate 3e-5 \
  --max-span-length 14 --negative-ratio 10 --seed 0 --output-dir output/span_conll2003_seed0
```

Further BERT experiments:

```bash
# Learning curve. Sizes 50, 100 and 0.01 train for 20 epochs, 0.1 and 1.0 for 3 epochs
python -B -m models.neural.bert_extended --task native --dataset wnut17 --train-size 50 --epochs 20 \
  --eval-every 5 --batch-size 8 --seed 0 --output-dir output/bert_wnut17_50_seed0
# Transfer on PER, LOC and ORG. SOURCE is conll2003, wnut17 or merged
python -B -m models.neural.bert_extended --task transfer --source merged --epochs 3 --batch-size 16 --learning-rate 2e-5 \
  --seed 0 --output-dir output/bert_transfer_merged_seed0
# Training on labels written by the LLM (section 7) or on gold labels of the same queries
python -B -m models.neural.bert_extended --task distill --label-source pseudo \
  --pseudo-predictions preds/mit_restaurant/qwen7b_k0_plain_train_2000_v1/0.jsonl \
  --epochs 3 --batch-size 8 --seed 0 --output-dir output/bert_distill_pseudo_seed0
# BiLSTM-CRF input representations. MODE is random, glove, glove_char or bert
python -B -m models.neural.bilstm_representations --mode glove --dataset conll2003 --train-size 1.0 --dev-size 0 \
  --epochs 5 --batch-size 16 --hidden-size 128 --learning-rate 0.001 --dropout 0.3 --seed 0 \
  --output-dir output/bilstm_repr_glove_seed0
```

Prediction and scoring:

```bash
# METHOD is bert, bilstm or span. Add --subset-ids data/subsets/conll2003/test_300.txt for the fixed subset.
python -B -m models.neural.predict_neural --method bert --model output/bert_conll2003_seed0 \
  --input data/processed/conll2003/test.jsonl --output preds/conll2003/bert_full/0.jsonl
# F1 with 95% bootstrap intervals and paired differences between methods
python -B -m models.neural.score_neural --gold data/processed/conll2003/test.jsonl \
  --pred M3=preds/conll2003/bilstm_full/0.jsonl --pred M4=preds/conll2003/bert_full/0.jsonl --pred Span=preds/conll2003/span_full/0.jsonl \
  --entity-types LOC MISC ORG PER --output results/b/paired_bootstrap_conll2003_test.json
# Boundary errors of the tagger against the span classifier
python -B -m models.neural.compare_boundary --gold data/processed/conll2003/test.jsonl \
  --pred M4=preds/conll2003/bert_full/0.jsonl --pred Span=preds/conll2003/span_full/0.jsonl \
  --output results/b/boundary_errors_conll2003_test.json
```

## 7. Zero-shot and LLM extraction

Environment: `requirements-llm.txt`. The models download from the Hugging Face Hub on first use: `urchade/gliner_medium-v2.1`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and `sentence-transformers/all-MiniLM-L6-v2`.

```bash
ALL="conll2003 conllpp wikigold btc wnut17 mit_restaurant"

# GLiNER on the six fixed subsets
python -B -m models.llm.gliner_zero

# Qwen2.5 with no examples
python -B -m models.llm.prompt_llm --model 1.5b --dataset $ALL --split test --subset test_300
python -B -m models.llm.prompt_llm --model 7b --dataset $ALL --split test --subset test_300

# Qwen2.5-7B with 10 examples retrieved from the training split
python -B -m models.llm.prompt_llm --model 7b --shots 10 --selection similar --dataset $ALL --split test --subset test_300
```

The other experiments change one option of the same script.

| Experiment | Options |
|---|---|
| Prompt selection on 300 CoNLL-2003 validation sentences | `--shots {0,1,5,10}`, `--selection {random,similar}`, `--guideline`, `--model {1.5b,7b}` with the default split and subset |
| Label budget | `--pool-size {50,100}` with `--seed {0,1,2}`, or `--pool-size {0.01,0.1}` |
| Instruction wording | `--wording {1,2}` |
| Annotation rules | `--guideline` on `conll2003` and `wnut17` |
| Full test set | `--dataset conll2003 --split test --subset full` |
| Edited test sets | `--dataset conll2003 --split test --input data/derived/perturbations/conll2003/lowercase_v1.jsonl --scope lowercase_v1` |
| Labels for distillation | `--model 7b --dataset mit_restaurant --train-sample 2000 --scope train_2000_v1` |

LoRA fine-tuning of the 1.5B model on 10% of CoNLL-2003:

```bash
python -B -m models.llm.lora_llm
python -B -m models.llm.prompt_llm --model 1.5b --adapter output/lora/qwen1.5b_conll2003_n0.1_seed0 \
  --dataset conll2003 conllpp wikigold --split test --subset test_300
python -B -m models.llm.prompt_llm --model 1.5b --adapter output/lora/qwen1.5b_conll2003_n0.1_seed0 \
  --dataset btc wnut17 --schema conll2003 --split test --subset test_300
```

The fine-tuned model reads the full test set and the edited test sets with the same `--adapter` option and the
options of the table above.

Each run writes its predictions and a score report under `results/c/`. Two CPU steps collect them.

```bash
python -B -m models.llm.error_types   # error types of the 7B model with and without annotation rules
python -B -m models.llm.summarize     # results/C.csv
```

## 8. Figures and tables

Environment: `requirements.txt`. The scripts read `results/` and write `report/figures/` and `report/tables/`. The figures use Times New Roman when it is installed.

```bash
python -B -m analysis.data_exploration
python -B -m analysis.make_figures        # F1, F3a, F5, F6b, F7
python -B -m analysis.make_figures_c      # F8, F9
python -B -m analysis.make_tables         # T1, T2, voting of the classic models
python -B -m analysis.make_tables_c       # zero-shot, label budget, wording, robustness, LoRA, distillation
python -B -m analysis.make_tables_team    # T3, T4 and T5 for all methods
# plot_scores and plot_entity need a new output path, so remove the old PDF first
python -B -m models.neural.plot_scores f2  --csv results/A.csv results/B.csv --output report/figures/f2.pdf
python -B -m models.neural.plot_scores f3b --csv results/B.csv --output report/figures/f3b.pdf
python -B -m models.neural.plot_scores f6a --csv results/B.csv --output report/figures/f6a.pdf
```

### All extractors together

`analysis.team_results` scores the ten extractors on the same sentences and writes `results/team_analysis.json`, which F5, F7, T4 and T5 read. It runs on CPU from prediction files and loads no model.

```bash
python -B -m analysis.team_results
```

It reads `preds/<dataset>/<method>_<scope>/0.jsonl`, where the scope is `full`, `lowercase_v1`, `spelling_noise_v1`, `no_punctuation_v1`, `replacement_original_v2` or `replacement_replaced_v2` on `conll2003`, and `test_300` on `conllpp`.

| Extractors | Method names | Written by |
|---|---|---|
| Dictionary, logistic regression, CRF | paths recorded in `results/m0_m2_*.json` | `run_baselines`, `run_temporal_baselines` and `run_robustness` of section 5 |
| BiLSTM-CRF, BERT, span classifier (seed 0) | `bilstm`, `bert`, `span` | `models.neural.predict_neural` of section 6, with `--input` set to the test file or to a file under `data/derived/` |
| GLiNER, Qwen2.5-7B, LoRA | `gliner`, `qwen7b_k0_plain`, `qwen7b_k10_similar_plain`, `lora1.5b_k0_plain` | `models.llm` of section 7 |

- **Seen and unseen entities (F5a):** the full CoNLL-2003 test set. An entity is seen when its words occur as a gold entity in the CoNLL-2003 training split.
- **Entity replacement (F5b):** the 500 sentence pairs of `data/derived/entity_replacement/`.
- **CoNLL-2003 against CoNLL++ (F5c):** the fixed 300 sentences of each.
- **Edited test sets (F7):** the full CoNLL-2003 test set and its three edited copies.
- **Voting (T4):** four groups of three extractors. A vote keeps an entity that at least two members return with the same span and type. The change is measured against the best member.
- **Error overlap (T4):** the Jaccard index of the missed and spurious entities of every pair of extractors.

Intervals use 1,000 sentence-bootstrap resamples, and paired differences resample the same sentences in both conditions.

The entity vector figure F4 needs the trained BERT tagger:

```bash
python -B -m models.neural.export_bert_entities --input data/processed/conll2003/validation.jsonl \
  --model bert-base-cased --output output/vectors/base_validation.jsonl
python -B -m models.neural.export_bert_entities --input data/processed/conll2003/validation.jsonl \
  --model output/bert_conll2003_seed0/best_model --fine-tuned --output output/vectors/finetuned_validation.jsonl
python -B -m models.neural.plot_entity --gold data/processed/conll2003/validation.jsonl \
  --glove data/external/glove/glove.6B.100d.txt --base-bert output/vectors/base_validation.jsonl \
  --fine-tuned-bert output/vectors/finetuned_validation.jsonl --max-samples 1000 --seed 42 \
  --output report/figures/f4.pdf --metrics results/b/f4_clustering.json
```

## 9. Entity extraction from raw text

Environment: `requirements.txt`, and `requirements-neural.txt` for `m4`.

`scripts/predict.py` returns the entities of a sentence with their type and character span, where the start is counted from 0 and the end is excluded. `m2` is the CRF and is fitted on each call; `m0` and `m1` select the dictionary and logistic regression. `m4` is the fine-tuned BERT tagger, the main extractor of the project, and needs one training run first. `--checkpoint` points it to another training output directory.

```bash
python -B -m scripts.predict --model m2 --text "John Smith visited London." --pos-python $POS

python -B -m models.neural.bert_baseline --dataset conll2003 --train-size 1.0 --dev-size 0 --epochs 3 \
  --batch-size 16 --learning-rate 2e-5 --output-dir output/bert_main
python -B -m scripts.predict --model m4 --text "John Smith visited London." --pos-python $POS
```
