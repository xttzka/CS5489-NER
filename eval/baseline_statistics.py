"""Score native NER baselines and compare them on paired bootstrap samples."""

import numpy as np

from eval.score_ner import bio_to_spans, score_ner


MODELS = ("m0", "m1", "m2")
COUNTS = ("tp", "fp", "fn")
METRICS = ("precision", "recall", "f1")


def compare(gold_by_id, pred_by_id):
    overall = score_ner(gold_by_id, pred_by_id)
    counts_by_id = {}
    for sample_id, gold_tags in gold_by_id.items():
        gold = bio_to_spans(gold_tags)
        pred = bio_to_spans(pred_by_id[sample_id])
        per_type = {}
        for metric, spans in zip(COUNTS, (gold & pred, pred - gold, gold - pred), strict=True):
            for _, _, kind in spans:
                counts = per_type.setdefault(kind, dict.fromkeys(COUNTS, 0))
                counts[metric] += 1
        counts_by_id[sample_id] = {
            **{metric: sum(row[metric] for row in per_type.values()) for metric in COUNTS},
            "per_type": per_type,
        }
    return overall, counts_by_id


def sentence_counts(gold_by_id, pred_by_id):
    """Return exact TP, FP and FN for each ID, with sparse per-type counts."""
    return compare(gold_by_id, pred_by_id)[1]


def metric_values(counts):
    tp, fp, fn = np.moveaxis(counts, -1, 0)

    def ratio(numerator, denominator):
        return np.divide(
            numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0,
        )

    return np.stack((ratio(tp, tp + fp), ratio(tp, tp + fn), ratio(2 * tp, 2 * tp + fp + fn)), axis=-1)


def bootstrap_counts(counts, resamples, seed):
    """Draw sentence multiplicities once per replicate for all model/type cells."""
    sentence_count = len(counts)
    flat = counts.reshape(sentence_count, -1)
    sampled = np.empty((resamples, flat.shape[1]), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for index in range(resamples):
        drawn = rng.integers(sentence_count, size=sentence_count)
        weights = np.bincount(drawn, minlength=sentence_count)
        sampled[index] = weights @ flat
    return sampled.reshape((resamples, *counts.shape[1:]))


def score_baselines(gold_by_id, predictions_by_model, *, resamples=1000, seed=42,
                    token_accuracy=False, entity_types=None):
    """Score M0–M2 on identical IDs, with paired 95% percentile intervals."""
    if set(predictions_by_model) != set(MODELS):
        raise ValueError("expected predictions for m0, m1 and m2")
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("resamples must be a positive integer")

    comparisons = {model: compare(gold_by_id, predictions_by_model[model]) for model in MODELS}
    observed_types = {
        kind for _, rows in comparisons.values() for row in rows.values() for kind in row["per_type"]
    }
    types = tuple(sorted(observed_types)) if entity_types is None else tuple(entity_types)
    if len(set(types)) != len(types) or observed_types.difference(types):
        raise ValueError("entity_types must be unique and include every observed type")

    ids = sorted(gold_by_id)
    counts = np.zeros((len(ids), len(MODELS), 1 + len(types), len(COUNTS)), dtype=np.int64)
    type_indices = {kind: index for index, kind in enumerate(types, start=1)}
    for model_index, model in enumerate(MODELS):
        _, rows = comparisons[model]
        for sentence_index, sample_id in enumerate(ids):
            row = rows[sample_id]
            counts[sentence_index, model_index, 0] = [row[metric] for metric in COUNTS]
            for kind, values in row["per_type"].items():
                counts[sentence_index, model_index, type_indices[kind]] = [values[metric] for metric in COUNTS]

    totals = counts.sum(axis=0)
    point_estimates = metric_values(totals)
    bootstrap = metric_values(bootstrap_counts(counts, resamples, seed))
    bounds = np.quantile(bootstrap, (0.025, 0.975), axis=0)

    def intervals(model_index, type_index):
        return {
            metric: bounds[:, model_index, type_index, metric_index].tolist()
            for metric_index, metric in enumerate(METRICS)
        }

    models = {}
    for index, model in enumerate(MODELS):
        overall, _ = comparisons[model]
        models[model] = {
            **overall,
            "bootstrap_ci95": intervals(index, 0),
            "per_type": {
                kind: {
                    "gold_records": len(ids),
                    "scored_records": len(ids),
                    **dict(zip(COUNTS, totals[index, type_index].tolist(), strict=True)),
                    **dict(zip(METRICS, point_estimates[index, type_index].tolist(), strict=True)),
                    "bootstrap_ci95": intervals(index, type_index),
                }
                for kind, type_index in type_indices.items()
            },
        }
        if token_accuracy:
            answer = predictions_by_model[model]
            total_tokens = sum(len(tags) for tags in gold_by_id.values())
            correct = sum(
                expected == predicted
                for sample_id, tags in gold_by_id.items()
                for expected, predicted in zip(tags, answer[sample_id], strict=True)
            )
            models[model].update(
                token_accuracy=correct / total_tokens if total_tokens else 0.0,
                correct_tokens=correct, scored_tokens=total_tokens,
            )

    paired = {}
    for first, second in ((0, 1), (1, 2)):
        point = models[MODELS[second]]["f1"] - models[MODELS[first]]["f1"]
        changes = bootstrap[:, second, 0, 2] - bootstrap[:, first, 0, 2]
        low, high = np.quantile(changes, (0.025, 0.975)).tolist()
        paired[f"{MODELS[second]}_minus_{MODELS[first]}"] = {
            "f1_difference": point,
            "ci95": [low, high],
            "f1_difference_pp": 100 * point,
            "ci95_pp": [100 * low, 100 * high],
        }
    return {
        "models": models,
        "paired_f1_differences": paired,
        "bootstrap": {"resamples": resamples, "seed": seed, "unit": "sentence", "interval": "95% percentile"},
    }
