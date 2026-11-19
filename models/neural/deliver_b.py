"""Verify completed v2 results and copy only small role-B handoff artifacts.

Dry-run by default. --run publishes new files only, never overwrites, and writes
a source/destination SHA-256 manifest. It does not copy datasets, checkpoints,
entity vectors, or prediction files. The original report's internal b_handoff
paths are preserved under report/b_handoff/; figures are also copied to uniquely
named files in report/figures/ for easy discovery.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
WORK_REL = Path('output/b_finish_v2')
LIMIT_BYTES = 25 * 1024 * 1024
REQUIRED_STEPS = frozenset({
    'plan', 'classic_plan', 'freeze', 'evaluate', 'compare_parent',
    'classic_predict_conll2003', 'classic_predict_wnut17', 'classic_predict_mit_restaurant',
    'classic_score', 'collect', 'f8', 'verify_c_t3', 'report', 'paired_models',
    'boundary_compare', 'additional_analysis', 'figure_f3b', 'figure_f6a', 'figure_f6b',
    'figure_f2', 'vectors_base', 'vectors_finetuned', 'vectors_glove', 'vector_inventory',
    'figure_f4', 'misc_clusters', 'misc_review_material',
})
FIGURES = ('f2_b.pdf', 'f3b_b.pdf', 'f6a_b.pdf', 'f6b_from_a_m2.pdf')
TABLES = ('t3_b_native_full.tex', 't3_b_300.tex', 't3_b_seed_summary.tex', 'b_sections.tex')
COPIES = (
    ('B.csv', 'results/B.csv'),
    ('classic_scores/classic_B.csv', 'results/B_classic_baselines.csv'),
    ('f8_handoff/f8_b_by_seed.csv', 'results/B_f8_by_seed.csv'),
    ('f8_handoff/f8_b_summary.csv', 'results/B_f8_summary.csv'),
    ('f8_handoff/f8_b_draft.md', 'results/B_f8_draft.md'),
    ('vectors/manifest.json', 'results/B_vectors_manifest.json'),
    ('vectors/glove_validation_metadata.json', 'results/B_glove_validation_metadata.json'),
    ('analysis/f6b_from_a_m2.json', 'results/B_f6b_from_a_m2.json'),
    ('analysis/paired_models_conll_seed0.json', 'results/B_paired_models_conll_seed0.json'),
    ('analysis/boundary_m4_span_conll_seed0.json', 'results/B_boundary_m4_span_conll_seed0.json'),
    ('version_comparison/summary.json', 'results/B_alignment_version_comparison.json'),
) + tuple((f'latex_handoff/{name}', f'report/b_handoff/{name}') for name in TABLES) \
  + tuple((f'latex_handoff/figures/{name}', f'report/b_handoff/figures/{name}') for name in FIGURES) \
  + tuple((f'latex_handoff/figures/{name}', f'report/figures/b_handoff_{name}') for name in FIGURES)
CHECKED_F4_COPIES = (
    ('output/b_f4_checked_v1/manifest.json', 'results/B_f4_checked_manifest.json'),
    ('output/b_f4_checked_v1/f4_openblas_single_thread_metrics.json', 'results/B_f4_metrics.json'),
    ('output/b_f4_checked_v1/f4_openblas_single_thread.pdf', 'report/b_handoff/figures/f4_b.pdf'),
    ('output/b_f4_checked_v1/f4_openblas_single_thread.pdf', 'report/figures/b_handoff_f4_b.pdf'),
)
SUPPLEMENT_COPIES = tuple(
    (f'models/neural/{name}', f'report/b_handoff/{name}') for name in (
        'B_methods_draft.tex', 'B_classic_methods_draft.tex',
        'B_results_analysis_draft.tex', 'B_misc_cluster_review.md')
)

README = '''# Role B handoff

This directory contains B's generated report fragment and tables. From the
`report/` directory, include `\\input{b_handoff/b_sections.tex}`. The fragment
requires the LaTeX `graphicx` and `booktabs` packages, and retains the generated
`b_handoff/` relative paths. It is a draft: resolve all TODOs and conduct the
MISC human review before submitting. F6(b) uses A's M2 overlap analysis and
the T3 300-sentence table includes compatible independently checked C rows.
The figures are also available in `report/figures/` with `b_handoff_` names.
The F4 PDF is the independently checked single-thread OpenBLAS redraw; its
audit manifest and metrics are `results/B_f4_checked_manifest.json` and
`results/B_f4_metrics.json`. The original v2 F4 remains under `output/`.

The independently reviewed `B_*` draft fragments and the preliminary
assistant-authored MISC worksheet are copied from `models/neural/`; their
source/destination hashes are included in the delivery manifest and they are
not outputs of the frozen finish journal. The results-analysis draft uses
`report/b_handoff/` paths relative to the repository root; check all TODOs and
obtain an independent human MISC review before submission.

`results/B.csv` contains neural evaluation records; the separately named
classic CSV records the reproduced M0--M2 baselines. Source files, SHA-256
digests, the frozen-manifest digest and journal digest are documented in
`results/B_delivery_manifest.json`. The vector manifest identifies large vector
files retained under `output/`; vectors, checkpoints, datasets and prediction
files are not included here. The original report's data and reproducibility
limitations still apply.
'''


def sha256(path):
    state = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            state.update(chunk)
    return state.hexdigest()


def resolve_inside(root, value):
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f'path escapes project: {value}')
    return resolved


def frozen_manifest(path):
    raw = json.loads(path.read_text(encoding='utf-8'))
    signed = dict(raw)
    recorded = signed.pop('manifest_sha256', None)
    serialized = json.dumps(signed, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    if (raw.get('version') != 1 or raw.get('state') != 'frozen' or raw.get('partial') is not False
            or raw.get('missing_requirements') != [] or hashlib.sha256(serialized).hexdigest() != recorded):
        raise ValueError('invalid, partial or changed frozen manifest')
    return raw


def read_journal(path):
    records = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        entry = json.loads(line)
        name = entry['step']
        if name in records:
            raise ValueError(f'duplicate finish journal step: {name}')
        records[name] = entry
    missing = REQUIRED_STEPS - records.keys()
    if missing:
        raise ValueError(f'finish workflow is incomplete; missing {len(missing)} steps: {", ".join(sorted(missing))}')
    return records


def check_journal(root, work, records):
    outputs = {}
    for name, entry in records.items():
        if not isinstance(entry.get('command'), list) or not isinstance(entry.get('outputs'), dict):
            raise ValueError(f'invalid finish record: {name}')
        for raw_path, expected in entry['outputs'].items():
            path = resolve_inside(root, raw_path)
            if not path.is_relative_to(work.resolve()) or not path.is_file() or sha256(path) != expected:
                raise ValueError(f'finish output missing, outside work dir or changed: {name}: {path}')
            if path in outputs and outputs[path] != expected:
                raise ValueError(f'conflicting journal output hashes: {path}')
            outputs[path] = expected
    if outputs.get((work/'frozen.json').resolve()) != sha256(work/'frozen.json'):
        raise ValueError('frozen manifest does not match the completed freeze journal step')
    return outputs


def check_evaluation(root, work, frozen):
    """Check task metadata and saved artifact hashes without opening test inputs."""
    tasks = frozen.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('frozen manifest has no evaluation tasks')
    seen = set()
    for task in tasks:
        task_id = task.get('id')
        if not isinstance(task_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', task_id) or task_id in seen:
            raise ValueError(f'invalid or duplicated evaluation task id: {task_id!r}')
        seen.add(task_id)
        directory = work/'evaluation'/'tasks'/task_id
        metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('task') != task or metadata.get('frozen_manifest_sha256') != frozen['manifest_sha256']:
            raise ValueError(f'{task_id}: task/manifest provenance mismatch')
        if (metadata.get('device') != 'mps' or metadata.get('resamples') != frozen['bootstrap']['resamples']
                or metadata.get('bootstrap_seed') != frozen['bootstrap']['seed']):
            raise ValueError(f'{task_id}: device or bootstrap provenance mismatch')
        run_key = str(resolve_inside(root, task['run_dir']))
        if metadata.get('checkpoint_sha256') != frozen['runs'][run_key]['checkpoint_sha256']:
            raise ValueError(f'{task_id}: checkpoint provenance mismatch')
        for field, hash_key in (('input', 'input_sha256'), ('subset_ids', 'subset_sha256')):
            if task[field] is None:
                if metadata.get(hash_key) is not None:
                    raise ValueError(f'{task_id}: unexpected {hash_key}')
                continue
            input_path = resolve_inside(root, task[field])
            binding_name = hashlib.sha256(str(input_path).encode()).hexdigest() + '.json'
            binding = json.loads((work/'evaluation'/'input_hashes'/binding_name).read_text(encoding='utf-8'))
            if binding != {'path': str(input_path), 'sha256': metadata[hash_key],
                           'frozen_manifest_sha256': frozen['manifest_sha256']}:
                raise ValueError(f'{task_id}: input-hash provenance mismatch')
        for filename, key in (('scores.json', 'scores_sha256'), ('predictions.jsonl', 'prediction_sha256')):
            if sha256(directory/filename) != metadata.get(key):
                raise ValueError(f'{task_id}: changed or missing {filename}')
        if metadata.get('profile_sha256') is not None and sha256(directory/'prediction_profile.json') != metadata['profile_sha256']:
            raise ValueError(f'{task_id}: changed prediction profile')
        score = json.loads((directory/'scores.json').read_text(encoding='utf-8'))
        if score.get('sentence_count') != metadata.get('sentences') or set(score.get('models', {})) != {task['method']}:
            raise ValueError(f'{task_id}: scores do not match prediction metadata')
        if score.get('bootstrap', {}).get('resamples') != frozen['bootstrap']['resamples'] or score.get('bootstrap', {}).get('seed') != frozen['bootstrap']['seed']:
            raise ValueError(f'{task_id}: score bootstrap differs from frozen protocol')
    return len(seen)


def check_csv(path):
    with path.open(encoding='utf-8', newline='') as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {'member', 'method', 'split', 'value'} <= set(reader.fieldnames):
            raise ValueError('B.csv does not have the expected result-table columns')
        count = 0
        for row in reader:
            if row['member'] != 'B' or row['split'] != 'test':
                raise ValueError('B.csv contains records outside B held-out evaluation')
            count += 1
    if not count:
        raise ValueError('B.csv is empty')
    return count


def check_f4(root, work, frozen, outputs):
    """Bind the corrected F4 redraw to the frozen v2 inputs and audit manifest."""
    audit_dir = root/'output/b_f4_checked_v1'
    manifest_path = audit_dir/'manifest.json'
    audit = json.loads(manifest_path.read_text(encoding='utf-8'))
    inputs = audit['inputs']
    expected = {
        'gold': root/'data/processed/conll2003/validation.jsonl',
        'glove': root/'data/external/glove/glove.6B.100d.txt',
        'base_bert': work/'vectors/base_validation.jsonl',
        'fine_tuned_bert': work/'vectors/finetuned_validation.jsonl',
        'original_figure': work/'latex_handoff/figures/f4_b.pdf',
        'original_metrics': work/'analysis/f4_metrics.json',
        'plot_code': root/'models/neural/plot_entity_b.py',
        'frozen_manifest': work/'frozen.json',
    }
    if set(inputs) != set(expected):
        raise ValueError('checked F4 audit input inventory differs from required sources')
    for label, path in expected.items():
        record = inputs[label]
        resolved = path.resolve()
        if (resolve_inside(root, record['path']) != resolved or not resolved.is_file()
                or sha256(resolved) != record['sha256']):
            raise ValueError(f'checked F4 source is missing or changed: {label}')
        if resolved.is_relative_to(work) and outputs.get(resolved) != record['sha256']:
            raise ValueError(f'checked F4 source differs from finish journal: {label}')
    if inputs['plot_code']['sha256'] != frozen['code_sha256']['models/neural/plot_entity_b.py']:
        raise ValueError('checked F4 plot code differs from frozen code')
    if (audit.get('runtime_warnings') != [] or audit.get('environment', {}).get('gpu_used') is not False
            or audit.get('parameters', {}).get('blas_threads') != 1):
        raise ValueError('checked F4 audit records warnings, GPU use, or different BLAS threading')
    actual_outputs = audit.get('outputs')
    checked_outputs = (audit_dir/'f4_openblas_single_thread.pdf',
                       audit_dir/'f4_openblas_single_thread_metrics.json',
                       audit_dir/'f4_openblas_single_thread.log')
    if not isinstance(actual_outputs, dict) or set(actual_outputs) != {str(path.resolve()) for path in checked_outputs}:
        raise ValueError('checked F4 output inventory differs from required redraw files')
    for path in checked_outputs:
        if not path.is_file() or sha256(path) != actual_outputs[str(path.resolve())]:
            raise ValueError(f'checked F4 output missing or changed: {path}')
    if (sha256(audit_dir/'f4_openblas_single_thread_metrics.json') != inputs['original_metrics']['sha256']
            or audit.get('script_sha256') != sha256(audit_dir/'regenerate.py')):
        raise ValueError('checked F4 metrics or redraw script differ from audit')
    for kind in ('GloVe', 'BERT (base)', 'BERT (fine-tuned)'):
        for metric in ('ari', 'nmi'):
            row = audit['ari_nmi_comparison'][kind][metric]
            if row['difference'] != 0.0 or row['original'] != row['checked']:
                raise ValueError(f'checked F4 clustering comparison differs: {kind}/{metric}')
    return {path.resolve(): sha256(path) for path in (manifest_path, *checked_outputs)}


def delivery_plan(root, work, outputs, checked_f4):
    items = []
    for source_rel, target_rel in COPIES:
        source = (work/source_rel).resolve()
        target = resolve_inside(root, target_rel)
        if not source.is_relative_to(work.resolve()) or source not in outputs:
            raise ValueError(f'delivery source is not a completed journal artifact: {source_rel}')
        if source.stat().st_size > LIMIT_BYTES:
            raise ValueError(f'refusing to copy large artifact: {source_rel}')
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'refusing to overwrite existing destination: {target_rel}')
        items.append({'source': source, 'destination': target, 'sha256': outputs[source], 'size': source.stat().st_size})
    for source_rel, target_rel in CHECKED_F4_COPIES:
        source = resolve_inside(root, source_rel)
        target = resolve_inside(root, target_rel)
        if source not in checked_f4 or not source.is_file() or sha256(source) != checked_f4[source]:
            raise ValueError(f'checked F4 artifact is missing or changed: {source_rel}')
        if source.stat().st_size > LIMIT_BYTES:
            raise ValueError(f'refusing to copy large artifact: {source_rel}')
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'refusing to overwrite existing destination: {target_rel}')
        items.append({'source': source, 'destination': target, 'sha256': checked_f4[source], 'size': source.stat().st_size})
    for source_rel, target_rel in SUPPLEMENT_COPIES:
        source = resolve_inside(root, source_rel)
        target = resolve_inside(root, target_rel)
        if not source.is_file() or source.stat().st_size > LIMIT_BYTES:
            raise ValueError(f'supplement is missing or unexpectedly large: {source_rel}')
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'refusing to overwrite existing destination: {target_rel}')
        items.append({'source': source, 'destination': target, 'sha256': sha256(source),
                      'size': source.stat().st_size})
    for target in (root/'report/b_handoff/README.md', root/'results/B_delivery_manifest.json'):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'refusing to overwrite existing destination: {target}')
    if len({item['destination'] for item in items}) != len(items):
        raise ValueError('multiple artifacts would write the same destination')
    return items


def publish(root, work, frozen, journal, items, task_count, row_count):
    """Stage on the same filesystem; hard-link each new target atomically."""
    published = []
    with tempfile.TemporaryDirectory(prefix='.b_delivery_', dir=work) as temporary:
        staging = Path(temporary)
        for index, item in enumerate(items):
            staged = staging/str(index)
            state = hashlib.sha256()
            with item['source'].open('rb') as source, staged.open('xb') as dest:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    state.update(chunk)
                    dest.write(chunk)
            if state.hexdigest() != item['sha256']:
                raise ValueError(f'source changed while staging: {item["source"]}')
            item['staged'] = staged
        readme = staging/'README.md'
        readme.write_text(README, encoding='utf-8')
        for item in items:
            item['destination'].parent.mkdir(parents=True, exist_ok=True)
            os.link(item['staged'], item['destination'])
            published.append(item['destination'])
        readme_target = root/'report/b_handoff/README.md'
        readme_target.parent.mkdir(parents=True, exist_ok=True)
        os.link(readme, readme_target)
        published.append(readme_target)
        records = [{'source': str(item['source'].relative_to(root)),
                    'destination': str(item['destination'].relative_to(root)),
                    'source_sha256': item['sha256'], 'destination_sha256': sha256(item['destination']),
                    'bytes': item['size']} for item in items]
        records.append({'source': 'generated by models/neural/deliver_b.py',
                        'destination': str(readme_target.relative_to(root)),
                        'source_sha256': sha256(readme), 'destination_sha256': sha256(readme_target),
                        'bytes': readme_target.stat().st_size})
        if any(record['source_sha256'] != record['destination_sha256'] for record in records):
            raise ValueError('a destination changed while delivery was being created')
        output = {'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  'frozen_manifest_sha256': frozen['manifest_sha256'], 'finish_journal_sha256': sha256(journal),
                  'evaluated_tasks': task_count, 'neural_csv_rows': row_count, 'artifacts': records,
                  'limitations': ['Report fragments may contain TODOs; manual MISC review is not claimed.',
                                  'Datasets, trained weights, entity vectors and complete predictions are excluded.']}
        target = root/'results/B_delivery_manifest.json'
        manifest_staged = staging/'delivery_manifest.json'
        with manifest_staged.open('x', encoding='utf-8') as dest:
            json.dump(output, dest, ensure_ascii=False, indent=2)
            dest.write('\n')
        os.link(manifest_staged, target)
        published.append(target)
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Publish only after every completion and integrity check')
    args = parser.parse_args()
    root, work = ROOT.resolve(), (ROOT/WORK_REL).resolve()
    lock_path = work/'finish.lock'
    try:
        if not lock_path.is_file():
            raise ValueError('v2 finish workflow has not started; finish.lock is missing')
        with lock_path.open('r+') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('v2 finish workflow is still running; delivery is blocked') from error
            try:
                journal = work/'finish_journal.jsonl'
                records = read_journal(journal)
                frozen = frozen_manifest(work/'frozen.json')
                outputs = check_journal(root, work, records)
                count = check_evaluation(root, work, frozen)
                rows = check_csv(work/'B.csv')
                checked_f4 = check_f4(root, work, frozen, outputs)
                items = delivery_plan(root, work, outputs, checked_f4)
                print(f'Validated {len(records)} finish steps, {count} frozen tasks and {rows} B CSV rows.')
                print(f'Planned {len(items)} copied files plus README and SHA-256 manifest.')
                if not args.run:
                    print('Dry-run only; no files copied. Add --run after reviewing the plan.')
                    return 0
                written = publish(root, work, frozen, journal, items, count, rows)
                print(f'Delivered {len(written)} files; manifest: results/B_delivery_manifest.json')
                return 0
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
