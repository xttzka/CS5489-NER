"""Build an F8 learning-curve handoff from frozen B.csv rows and training data.

Only training splits, configs, a frozen manifest and already-aggregated CSV are
read. No evaluation dataset or prediction is opened, and missing values remain
blank. Reference sample IDs are reconstructed using the team's sample_train.
"""
import argparse
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

from eval.result_tables import read_csv
from models.llm.common import TYPE_DESCRIPTIONS, sample_train
from models.neural.run_b_evaluation import load_frozen, sha256

DATASETS = ('conll2003', 'wnut17', 'mit_restaurant')
BUDGETS = (('50', (0, 1, 2)), ('100', (0, 1, 2)), ('0.01', (0,)), ('0.1', (0,)), ('1.0', (0,)))
FIELDS = ('dataset', 'budget', 'seed', 'train_sentences', 'reference_train_ids_sha256',
          'training_ids_verification', 'method', 'split', 'scope', 'label_space', 'test_subset_sha256',
          'f1', 'status', 'task_id', 'frozen_manifest_sha256', 'source_ref')
SUMMARY_FIELDS = ('dataset', 'budget', 'train_sentences', 'completed_seeds', 'expected_seeds',
                  'mean_f1', 'sample_sd_f1', 'status', 'same_count_budgets')


def size_from_condition(condition):
    match = re.fullmatch(r'size=(50|100|0\.01|0\.1|1\.0);epochs=(\d+);eval_every=(\d+)', condition)
    if not match:
        raise ValueError(f'Unsupported learning-curve condition: {condition}')
    size, epochs, every = match.groups()
    expected = ('3', '1') if size in ('0.1', '1.0') else ('20', '5')
    if (epochs, every) != expected:
        raise ValueError(f'Learning schedule differs from the frozen protocol: {condition}')
    return size


def reference_sample(dataset, budget, seed, run_dir):
    rows = sample_train(dataset, budget, seed)
    ids = [row['sample_id'] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f'Invalid reconstructed training IDs for {dataset}/{budget}/seed{seed}')
    digest = hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
    file = Path(run_dir) / 'train_ids.txt'
    status = 'reconstructed_unverified'
    if file.exists():
        saved = [line.strip() for line in file.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(saved) != len(set(saved)) or set(saved) != set(ids):
            raise ValueError(f'Saved training IDs disagree with reference sampler: {file}')
        status = 'matches_saved_train_ids'
    return len(ids), digest, status


def build_rows(frozen, csv_rows):
    if frozen.get('state') != 'frozen' or frozen.get('partial'):
        raise ValueError('a complete frozen protocol is required')
    tasks = {}
    for task in frozen['tasks']:
        if task['group'] != 'learning':
            continue
        if task['target'] not in DATASETS or task['method'] != 'M4' or task['scope'] != 'test_300' or task['train_source'] != task['target']:
            raise ValueError(f"Invalid F8 task: {task['id']}")
        size = size_from_condition(task['condition'])
        key = (task['target'], size, task['seed'])
        if key in tasks:
            raise ValueError(f'Duplicate frozen learning task: {key}')
        tasks[key] = task
    expected = {(dataset, size, seed) for dataset in DATASETS for size, seeds in BUDGETS for seed in seeds}
    if set(tasks) != expected:
        raise ValueError(f'Frozen learning inventory differs: missing={sorted(expected-set(tasks))}, extra={sorted(set(tasks)-expected)}')

    scores = {}
    for row in csv_rows:
        if not (row['experiment'] == 'learning_curve' and row['method'] == 'M4' and row['split'] == 'test'
                and row['scope'] == 'test_300' and row['metric'] == 'f1' and row['entity_type'] == 'all'
                and row['entity_group'] == 'all'):
            continue
        dataset = row['dataset']
        if dataset not in DATASETS or row['train_source'] != dataset:
            continue
        size = size_from_condition(row['condition'])
        key = (dataset, size, int(row['seed']))
        if key not in tasks:
            raise ValueError(f'Unexpected learning-curve CSV row: {key}')
        if row['label_space'] != '+'.join(sorted(TYPE_DESCRIPTIONS[dataset])):
            raise ValueError(f'Wrong native label schema: {key}')
        config = json.loads(row['config_json'])
        if config.get('task_id') != tasks[key]['id'] or config.get('frozen_manifest_sha256') != frozen['manifest_sha256']:
            raise ValueError(f'CSV provenance does not match frozen task: {key}')
        if key in scores:
            raise ValueError(f'Duplicate learning-curve F1 row: {key}')
        value = float(row['value'])
        if not 0 <= value <= 1:
            raise ValueError(f'F1 outside [0,1]: {key}')
        source_path = Path(row['source_ref'].split('#', 1)[0])
        if source_path.name != 'scores.json':
            raise ValueError(f'Expected collector score source: {key}')
        metadata = json.loads((source_path.parent / 'metadata.json').read_text(encoding='utf-8'))
        if (metadata.get('task') != tasks[key] or metadata.get('frozen_manifest_sha256') != frozen['manifest_sha256']
                or metadata.get('scores_sha256') != sha256(source_path) or metadata.get('sentences') != 300
                or not metadata.get('subset_sha256')):
            raise ValueError(f'Evaluation artifact provenance differs: {key}')
        scores[key] = (value, row['source_ref'], metadata['subset_sha256'])

    for dataset in DATASETS:
        subset_hashes = {value[2] for key, value in scores.items() if key[0] == dataset}
        if len(subset_hashes) > 1:
            raise ValueError(f'Learning results use different fixed subsets: {dataset}')

    output = []
    for dataset, size, seed in sorted(expected):
        task = tasks[dataset, size, seed]
        run_dir = Path(task['run_dir']).resolve()
        frozen_run = frozen['runs'][str(run_dir)]
        config_path = run_dir / 'run_config.json'
        if sha256(config_path) != frozen_run['run_config_sha256']:
            raise ValueError(f'Frozen training config changed: {config_path}')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if str(config.get('train_size')) != size or config.get('seed') != seed or config.get('dataset') != dataset:
            raise ValueError(f'Frozen training config does not match task: {task["id"]}')
        count, digest, verification = reference_sample(dataset, size, seed, run_dir)
        result = scores.get((dataset, size, seed))
        output.append({'dataset': dataset, 'budget': size, 'seed': seed, 'train_sentences': count,
                       'reference_train_ids_sha256': digest, 'training_ids_verification': verification,
                       'method': 'M4', 'split': 'test', 'scope': 'test_300',
                       'label_space': '+'.join(sorted(TYPE_DESCRIPTIONS[dataset])),
                       'test_subset_sha256': result[2] if result else '',
                       'f1': result[0] if result else '', 'status': 'complete' if result else 'missing',
                       'task_id': task['id'], 'frozen_manifest_sha256': frozen['manifest_sha256'],
                       'source_ref': result[1] if result else ''})
    return sorted(output, key=lambda row: (row['dataset'], row['train_sentences'], row['budget'], row['seed']))


def summarize(rows):
    result = []
    for dataset in DATASETS:
        subset = [row for row in rows if row['dataset'] == dataset]
        for budget, seeds in BUDGETS:
            group = [row for row in subset if row['budget'] == budget]
            counts = {row['train_sentences'] for row in group}
            if len(counts) != 1:
                raise ValueError(f'Training count differs across seeds for {dataset}/{budget}')
            count = counts.pop()
            # Equal x coordinates remain separate budget series, never pooled.
            collisions = sorted({row['budget'] for row in subset if row['train_sentences'] == count and row['budget'] != budget})
            values = [row['f1'] for row in group if row['status'] == 'complete']
            result.append({'dataset': dataset, 'budget': budget, 'train_sentences': count,
                           'completed_seeds': len(values), 'expected_seeds': len(seeds),
                           'mean_f1': statistics.mean(values) if values else '',
                           'sample_sd_f1': statistics.stdev(values) if len(values) > 1 else '',
                           'status': 'complete' if len(values) == len(seeds) else 'missing_or_partial',
                           'same_count_budgets': ','.join(collisions)})
    return sorted(result, key=lambda row: (row['dataset'], row['train_sentences'], row['budget']))


def write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True, help='B.csv produced by collect_b_evaluations')
    parser.add_argument('--output-dir', type=Path, required=True, help='New handoff directory')
    args = parser.parse_args()
    try:
        frozen = load_frozen(args.frozen)
        rows = build_rows(frozen, list(read_csv(args.csv)))
        summaries = summarize(rows)
        if args.output_dir.exists():
            raise FileExistsError(f'refusing to overwrite {args.output_dir}')
        args.output_dir.mkdir(parents=True)
        write_csv(args.output_dir / 'f8_b_by_seed.csv', FIELDS, rows)
        write_csv(args.output_dir / 'f8_b_summary.csv', SUMMARY_FIELDS, summaries)
        missing = sum(row['status'] == 'missing' for row in rows)
        report = [
            '# F8 BERT learning-curve handoff (draft)', '',
            'This draft uses only frozen held-out test_300 rows; validation metrics are excluded.',
            'Scores are entity-level micro F1 on native labels, on a 0–1 scale. The x-axis is actual sampled sentence count.',
            '50 and 100 use three seeds; 1%, 10% and 100% use seed 0. Partial averages are labelled, and no SD is calculated for a single seed.',
            'Each budget remains separate when two budgets happen to have the same sentence count. Do not pool duplicate x coordinates.',
            'Sampler IDs are reconstructed from training data; only rows marked matches_saved_train_ids are independently verified against saved IDs. The subset hash is recorded for C to cross-check.',
            '50/100/1% used 20 epochs with validation every 5 epochs, whereas 10%/100% used 3 epochs. This exploratory schedule differs by budget and affects interpretation.',
            '', f'Completed seed-level points: {len(rows)-missing}/{len(rows)}. Missing points remain blank.', '',
            '| Dataset | Budget | Training sentences | Completed/expected seeds | Test F1 mean | Sample SD | Status |',
            '|---|---:|---:|---:|---:|---:|---|']
        for item in summaries:
            mean = f"{item['mean_f1']:.4f}" if item['mean_f1'] != '' else 'TODO'
            sd = f"{item['sample_sd_f1']:.4f}" if item['sample_sd_f1'] != '' else '—'
            report.append(f"| {item['dataset']} | {item['budget']} | {item['train_sentences']} | {item['completed_seeds']}/{item['expected_seeds']} | {mean} | {sd} | {item['status']} |")
        report += ['', 'TODO: C should verify identical training sample-ID hashes and fixed test IDs before comparing methods.',
                   'TODO: Insert the combined F8 figure and interpret domain differences only after missing results are complete.', '']
        (args.output_dir / 'f8_b_draft.md').write_text('\n'.join(report), encoding='utf-8')
        print(f'Wrote handoff to {args.output_dir}; missing seed-level points: {missing}/{len(rows)}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
