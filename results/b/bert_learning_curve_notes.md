# F8 BERT learning-curve handoff (draft)

This draft uses only frozen held-out test_300 rows; validation metrics are excluded.
Scores are entity-level micro F1 on native labels, on a 0–1 scale. The x-axis is actual sampled sentence count.
50 and 100 use three seeds; 1%, 10% and 100% use seed 0. Partial averages are labelled, and no SD is calculated for a single seed.
Each budget remains separate when two budgets happen to have the same sentence count. Do not pool duplicate x coordinates.
Sampler IDs are reconstructed from training data; only rows marked matches_saved_train_ids are independently verified against saved IDs. The subset hash is recorded for C to cross-check.
50/100/1% used 20 epochs with validation every 5 epochs, whereas 10%/100% used 3 epochs. This exploratory schedule differs by budget and affects interpretation.

Completed seed-level points: 27/27. Missing points remain blank.

| Dataset | Budget | Training sentences | Completed/expected seeds | Test F1 mean | Sample SD | Status |
|---|---:|---:|---:|---:|---:|---|
| conll2003 | 50 | 50 | 3/3 | 0.5671 | 0.0233 | complete |
| conll2003 | 100 | 100 | 3/3 | 0.6837 | 0.0230 | complete |
| conll2003 | 0.01 | 140 | 1/1 | 0.7349 | — | complete |
| conll2003 | 0.1 | 1404 | 1/1 | 0.8513 | — | complete |
| conll2003 | 1.0 | 14041 | 1/1 | 0.8801 | — | complete |
| mit_restaurant | 50 | 50 | 3/3 | 0.3895 | 0.0446 | complete |
| mit_restaurant | 0.01 | 69 | 1/1 | 0.4646 | — | complete |
| mit_restaurant | 100 | 100 | 3/3 | 0.4986 | 0.0222 | complete |
| mit_restaurant | 0.1 | 689 | 1/1 | 0.6725 | — | complete |
| mit_restaurant | 1.0 | 6894 | 1/1 | 0.7806 | — | complete |
| wnut17 | 0.01 | 34 | 1/1 | 0.1543 | — | complete |
| wnut17 | 50 | 50 | 3/3 | 0.1298 | 0.0929 | complete |
| wnut17 | 100 | 100 | 3/3 | 0.2344 | 0.0762 | complete |
| wnut17 | 0.1 | 339 | 1/1 | 0.1662 | — | complete |
| wnut17 | 1.0 | 3394 | 1/1 | 0.4479 | — | complete |

TODO: C should verify identical training sample-ID hashes and fixed test IDs before comparing methods.
TODO: Insert the combined F8 figure and interpret domain differences only after missing results are complete.
