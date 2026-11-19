"""Collect verified frozen B evaluation artifacts into team CSV without re-reading test data.

No model execution or scoring occurs here. Existing semantic records with the
same value are skipped; conflicting values abort without changing the CSV.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from eval.result_tables import read_csv, validate_rows
from models.neural.collect_b_results import semantic_key
from models.neural.run_b_evaluation import load_frozen, read_json, verify_result
from models.neural.score_b import append_csv, output_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    args = parser.parse_args()
    try:
        frozen = load_frozen(args.frozen)
        old = list(read_csv(args.csv)) if args.csv.exists() else []
        existing = {semantic_key(row): row for row in old}
        if len(existing) != len(old):
            raise ValueError('existing CSV contains duplicate semantic metric keys')
        pending, missing, completed = [], [], 0
        for task in frozen['tasks']:
            directory = args.output_root / 'tasks' / task['id']
            if not directory.exists():
                missing.append(task['id'])
                continue
            metadata = read_json(directory / 'metadata.json')
            run = frozen['runs'][str(Path(task['run_dir']).resolve())]
            verify_result(directory, task, frozen['manifest_sha256'], metadata['input_sha256'],
                          metadata['subset_sha256'], run['checkpoint_sha256'],
                          frozen['bootstrap']['resamples'], frozen['bootstrap']['seed'])
            if metadata.get('device') != 'mps':
                raise ValueError(f"{task['id']}: expected MPS prediction")
            for field, digest in (('input', metadata['input_sha256']), ('subset_ids', metadata['subset_sha256'])):
                if task[field] is None:
                    if digest is not None:
                        raise ValueError(f"{task['id']}: unexpected subset hash")
                    continue
                input_path = (Path(__file__).resolve().parents[2] / task[field]).resolve()
                name = hashlib.sha256(str(input_path).encode()).hexdigest() + '.json'
                binding = read_json(args.output_root / 'input_hashes' / name)
                if binding != {'path': str(input_path), 'sha256': digest,
                               'frozen_manifest_sha256': frozen['manifest_sha256']}:
                    raise ValueError(f"{task['id']}: input provenance binding differs")
            scores = read_json(directory / 'scores.json')
            if scores['sentence_count'] != metadata['sentences']:
                raise ValueError(f"{task['id']}: sentence count differs from prediction metadata")
            expected_types = ['LOC', 'ORG', 'PER'] if task['target_schema'] else task['entity_types']
            if set(scores['entity_types']) != set(expected_types):
                raise ValueError(f"{task['id']}: entity schema differs from frozen task")
            evaluation = scores['evaluation']
            expected = {'split': 'test', 'scope': task['scope'], 'target': task['target'],
                        'train_source': task['train_source'], 'experiment': task['experiment'],
                        'condition': task['condition'], 'seed': task['seed'],
                        'target_schema': task['target_schema'], 'prediction_schema': task['prediction_schema']}
            if evaluation != expected or set(scores['models']) != {task['method']}:
                raise ValueError(f"{task['id']}: scores disagree with frozen task")
            if scores['bootstrap']['resamples'] != frozen['bootstrap']['resamples'] or scores['bootstrap']['seed'] != frozen['bootstrap']['seed']:
                raise ValueError(f"{task['id']}: scores use a different bootstrap protocol")
            context = SimpleNamespace(experiment=task['experiment'], condition=task['condition'],
                                      train_source=task['train_source'], dataset=task['target'],
                                      split='test', scope=task['scope'], model_seed=task['seed'],
                                      scoring='transfer_three_types' if task['target_schema'] else 'native')
            for row in output_rows(scores, context, directory / 'scores.json'):
                row['config_json'] = json.dumps({'task_id': task['id'], 'frozen_manifest_sha256': frozen['manifest_sha256'],
                                                 'checkpoint_sha256': run['checkpoint_sha256'], 'bootstrap': scores['bootstrap'],
                                                 'scoring': context.scoring}, sort_keys=True, separators=(',', ':'))
                key = semantic_key(row)
                prior = existing.get(key)
                if prior is not None:
                    if float(prior['value']) != float(row['value']) or prior['unit'] != row['unit'] or prior['config_json'] != row['config_json']:
                        raise ValueError(f"conflicting existing CSV metric: {key}")
                else:
                    existing[key] = row
                    pending.append(row)
            completed += 1
        list(validate_rows([*old, *pending]))
        if pending:
            append_csv(args.csv, pending)
        print(f'Verified tasks: {completed}; missing tasks: {len(missing)}; appended rows: {len(pending)}')
        for task_id in missing:
            print(f'MISSING {task_id}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
