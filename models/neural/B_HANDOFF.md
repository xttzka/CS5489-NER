# Role B handoff

The deliverables are the verified model checkpoints, task-level predictions and scores, the B results CSV, training/configuration evidence, and the F8 tables. A task or score is not complete until its saved metadata passes the frozen-protocol checks. Keep validation and held-out test results in separate rows and report sections.

The local frozen v2 evaluation has completed. The organized results and report are `results/b/` and `report/b/`; `results/B.csv` remains the team's canonical compatibility table. Local preparation is not proof of an actual handoff to A or C. The shared `scripts/predict.py` still defaults to M2 and needs team integration; the separate B primary-model entry was tested on this Mac only.

## F8 learning curves

The completed seed and summary files are `results/b/bert_learning_curve_by_seed.csv`, `results/b/bert_learning_curve_summary.csv`, and `results/b/bert_learning_curve_notes.md`. To rebuild them after restoring the required local artifacts, run:

```sh
conda activate cs5489
python -B -m models.neural.handoff_f8_b \
  --frozen PATH/TO/frozen.json \
  --csv PATH/TO/B.csv \
  --output-dir results/b_f8_handoff_v1
```

The output directory must not exist. The program reads training splits, frozen training configs, the CSV, and the saved scores/metadata. It never reads a held-out dataset or calls the GPU. It writes `f8_b_by_seed.csv`, `f8_b_summary.csv`, and `f8_b_draft.md`; missing results remain blank. Keep the seed-level table when combining B with C. The summary labels incomplete groups and calculates sample standard deviation only with at least two available seeds.

Plot the x-axis using `train_sentences`, in numeric order within each dataset. The 1% budget may lie before or between 50 and 100 depending on the dataset. If different budgets have the same actual count, retain separate points and disclose the coincidence: do not merge their scores. `reference_train_ids_sha256` is calculated from the team's `sample_train` rule and training split; `matches_saved_train_ids` means it was checked against a saved ID file, while `reconstructed_unverified` cannot by itself prove the original run used those IDs. Compare these hashes and `test_subset_sha256` with C before claiming matched samples. The hash is computed over sorted sample IDs joined with newline characters, without a trailing newline.

The current exploratory schedule uses 20 epochs with evaluation every 5 for 50, 100 and 1%, and 3 epochs for 10% and 100%; this difference limits a simple data-size-only interpretation. A single seed cannot support a seed-level standard deviation. F8 compares test results with compatible test results, never validation with test.

## Transferring artifacts

The repository `.gitignore` excludes `/output/`, almost all `/preds/`, `/data/processed/`, `/data/derived/`, and model files including `*.pt` and `*.safetensors`. A Git commit therefore will **not** normally contain the checkpoints, most prediction files, or processed inputs. `results/B.csv` and the F8 tables may be trackable, but they cannot reproduce an evaluation without the ignored artifacts and their provenance. Check exactly what is included with `git status --short` and `git check-ignore -v PATH`.

Share the frozen manifest, complete evaluation task directories (including metadata, predictions, scores and profiles where available), input-hash records, training configs and completion logs, checkpoints/tokenizers, and any permitted datasets through a mutually agreed storage channel. Source references and the frozen manifest include absolute paths from the original Mac. On another machine, preserve the original signed manifest and make a separately auditable path remapping or reconstruction; do not silently rewrite it. Follow the datasets' terms and avoid publishing restricted corpora or large checkpoints without checking distribution rights. For long-term reproducibility, include code revision, environment versions, validated data preparation instructions, and checksums. Do not silently alter `.gitignore` or force-add large files.

## Integrating the English LaTeX draft

The organized B report files are `report/b/role_b_report.tex`, `report/b/sections/`, `report/b/tables/` and `report/b/figures/`; compile the standalone file from the repository root. See `report/b/README.md` for the fragments and `booktabs`/`graphicx` dependencies. The source refers to those paths, but a separate joined team report still requires A/C integration and bibliography. The F4 scatter is a deterministic 1,000-entity validation subsample; the separate MISC review uses all matched validation entities. Its assistant-prepared sample review is not independent human annotation.

The current hyperparameter search changes candidate settings jointly. Any winner comparison is descriptive and must not be attributed to one individual hyperparameter. The F8 experiments use their separately fixed exploratory schedule; compare their data budgets only with this limitation disclosed.
