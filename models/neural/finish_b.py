"""Resume the full role-B post-training workflow after validation-only selection.

Dry-run by default. --run waits for the shared MPS lock, constructs a complete
validation-only selection, freezes the protocol and executes the already
implemented evaluation. Steps never overwrite existing deliverables. Outputs
are journalled with hashes; an interrupted evaluation is safely resumed by its
own task-level checks. Existing A/C test results do not participate in selection.
"""
import argparse
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from models.neural.run_b_evaluation import load_frozen, make_plan, read_json, sha256, write_new_json

ROOT = Path(__file__).resolve().parents[2]


def wait_for_queue(poll_seconds):
    path = ROOT / 'output/b_training_queue/queue.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as lock:
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'Training/evaluation queue busy; checking again in {poll_seconds:g}s.', flush=True)
                time.sleep(poll_seconds)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                return


def queue_processes():
    """Return active queue commands, or None when OS process listing is unavailable."""
    try:
        result = subprocess.run(['/bin/ps', '-axo', 'pid=,command='], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    needles = ('models.neural.run_b_training', 'models.neural.run_b_tuning', 'models.neural.run_b_representations')
    return [line.strip() for line in result.stdout.splitlines() if any(needle in line for needle in needles)]


def readiness(search_path, classic_evidence):
    """Full read-only train/dev audit. Caller owns the shared queue lock."""
    from models.neural.run_b_training import make_tasks as training_tasks, find_completed as training_done
    from models.neural.run_b_representations import make_tasks as representation_tasks, parse_args, find_completed as representation_done
    training = training_tasks()
    representations = representation_tasks(parse_args(['--device', 'mps']))
    if len(training) != 37 or len(representations) != 12:
        raise ValueError(f'unexpected required queue sizes: {len(training)} training, {len(representations)} representation')
    missing = [task.name for task in training if training_done(task)[0] is None]
    missing += [task.name for task in representations if representation_done(task)[0] is None]
    if not Path(search_path).is_file():
        missing.append('output/b_tuning/selection.json')
    if not Path(classic_evidence).is_file():
        missing.append('classic validation evidence')
    if missing:
        return missing
    # Non-partial planning independently rechecks every selected and candidate run,
    # MPS, dev scores, all markers, learning curves, transfer and distillation.
    draft = make_plan(ROOT, False, search_path)
    selection_evidence(draft, classic_evidence)
    return []


def queue_failure(directory):
    """Only the newest journal for each queue can declare a terminal failure."""
    for prefix in ('run_', 'tuning_', 'representations_'):
        journals = sorted(directory.glob(prefix+'*.jsonl'), key=lambda path: path.stat().st_mtime_ns)
        if not journals:
            continue
        events = []
        for line in journals[-1].read_text(encoding='utf-8').splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError(f'incomplete or corrupt queue journal: {journals[-1]}')
        for event in reversed(events):
            kind = event.get('event')
            if kind in ('queue_error', 'queue_interrupted'):
                return f'{journals[-1]}: {kind}: {event.get("error", "interrupted")}'
            if kind == 'queue_end':
                if event.get('failed', 0) or event.get('stopped_early'):
                    return f'{journals[-1]}: queue ended with failed or unfinished tasks'
                break
            if kind == 'queue_complete':
                break
    return None


def wait_until_ready(search_path, classic_evidence, poll_seconds, idle_grace):
    queue = ROOT/'output/b_training_queue/queue.lock'
    queue.parent.mkdir(parents=True, exist_ok=True)
    idle_since = None
    while True:
        with queue.open('a+') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                idle_since = None
                print('A training/tuning/representation queue holds the shared lock; waiting.', flush=True)
            else:
                try:
                    missing = readiness(search_path, classic_evidence)
                    failure = queue_failure(queue.parent) if missing else None
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                if not missing:
                    print('All 37 training runs, 12 representation markers and dev search passed locked audit.', flush=True)
                    return
                active = queue_processes()
                print(f'Not ready ({len(missing)} missing; e.g. {missing[0]}). Active queue processes: '
                      + ('unknown' if active is None else str(len(active))), flush=True)
                if active:
                    idle_since = None
                else:
                    if active == [] and failure:
                        raise RuntimeError(f'upstream queue failed and no queue process is running: {failure}')
                    idle_since = time.monotonic() if idle_since is None else idle_since
                    if time.monotonic() - idle_since >= idle_grace:
                        reason = 'no queue process is running' if active == [] else 'process listing is unavailable and the queue is unlocked'
                        raise RuntimeError(f'{reason}; training prerequisites remain incomplete: {missing[0]}' + (f'; latest queue failure: {failure}' if failure else ''))
        time.sleep(poll_seconds)


def load_journal(path):
    entries = {}
    if path.exists():
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            try:
                entry = json.loads(line)
                name = entry['step']
                if name in entries:
                    raise ValueError(f'duplicate completed step {name}')
                entries[name] = entry
            except (ValueError, KeyError) as error:
                raise ValueError(f'invalid finish journal at line {number}: {error}') from error
    return entries


def artifact_hashes(paths):
    result = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f'expected output missing: {path}')
        result[str(path.resolve())] = sha256(path)
    return result


def append_event(journal, entry):
    with journal.open('a', encoding='utf-8') as target:
        target.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + '\n')
        target.flush()
        os.fsync(target.fileno())


def run_step(name, command, outputs, work, journal, done, *, resume_output=False, use_mps_lock=False, recheck=False):
    command = [str(part) for part in command]
    if name in done:
        if done[name]['command'] != command or artifact_hashes(outputs) != done[name]['outputs']:
            raise ValueError(f'completed {name} output or command changed; refusing to resume')
        if not recheck:
            print(f'Already complete: {name}', flush=True)
            return
    if not resume_output and any(Path(path).exists() for path in outputs):
        raise FileExistsError(f'{name}: output exists without a completion record; preserve it and use another work directory')
    logs = work / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    number = 1
    while (logs / f'{name}_{number}.log').exists():
        number += 1
    log_path = logs / f'{name}_{number}.log'
    print(f'[{name}] {shlex.join(command)}\nLog: {log_path}', flush=True)
    env = os.environ.copy()
    env.update({'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'MPLCONFIGDIR': str(work / 'matplotlib_cache')})
    env.pop('PYTORCH_ENABLE_MPS_FALLBACK', None)
    lock = None
    try:
        if use_mps_lock:
            lock = (ROOT / 'output/b_training_queue/queue.lock').open('a+')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with log_path.open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if code:
            raise RuntimeError(f'{name} failed with exit {code}; see {log_path}')
    finally:
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
    if name in done:
        print(f'Rechecked: {name}', flush=True)
        return
    entry = {'step': name, 'command': command, 'outputs': artifact_hashes(outputs),
             'completed_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'log': str(log_path)}
    append_event(journal, entry)
    done[name] = entry


def selection_evidence(plan, classic_path):
    classic = read_json(classic_path)
    required = {'dataset': 'conll2003', 'split': 'validation', 'scope': 'full', 'label_space': 'LOC+MISC+ORG+PER'}
    if any(classic.get(key) != value for key, value in required.items()):
        raise ValueError('classic evidence is not the required full native-four-class validation')
    note = classic.get('classic_configuration_note')
    if not isinstance(note, str) or not note.strip():
        raise ValueError('classic evidence lacks a configuration note')
    classic_predictions = classic.get('predictions', [])
    if {(row.get('method'), row.get('seed')) for row in classic_predictions} != {(m, 0) for m in ('M0', 'M1', 'M2')} or len(classic_predictions) != 3:
        raise ValueError('classic evidence must have exactly M0/M1/M2 seed 0')
    entries = []
    for row in classic_predictions:
        path = Path(row['path']).resolve()
        digest = sha256(path)
        if row.get('sha256') != digest:
            raise ValueError(f'classic validation prediction hash mismatch: {path}')
        entries.append({'method': row['method'], 'seed': 0, 'path': str(path), 'sha256': digest})
    for method in ('M3', 'M4', 'Span'):
        for seed in range(3):
            matches = [task for task in plan['tasks'] if task['group'] == 'domain' and task['method'] == method
                       and task['target'] == 'conll2003' and task['scope'] == 'full' and task['seed'] == seed]
            if len(matches) != 1:
                raise ValueError(f'missing unique neural validation run: {method} seed{seed}')
            path = Path(matches[0]['run_dir']).resolve() / 'validation_predictions.jsonl'
            entries.append({'method': method, 'seed': seed, 'path': str(path), 'sha256': sha256(path)})
    return {**required, 'classic_configuration_note': note + ' This evidence reproduces the selected classic features and hyperparameters; prior A/C held-out results were already present in the repository before this automation.',
            'predictions': entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'output/b_finish_v1')
    parser.add_argument('--classic-evidence', type=Path, default=ROOT / 'output/classic_validation_b_v1/classic_evidence.json')
    parser.add_argument('--search-evidence', type=Path, default=ROOT / 'output/b_tuning/selection.json')
    parser.add_argument('--amend-from', type=Path, help='Previous frozen manifest to amend for tokenizer alignment')
    parser.add_argument('--alignment-audit', type=Path, help='Complete token-only audit bound to the parent manifest')
    parser.add_argument('--parent-evaluation-root', type=Path, help='Parent results for comparing all completed tasks')
    parser.add_argument('--classic-python', type=Path, default=ROOT / '.venv-b-classic/bin/python')
    parser.add_argument('--plot-python', type=Path, default=Path('/opt/anaconda3/bin/python3'))
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--wait-until-ready', action='store_true',
                        help='Wait for the 37-run, tuning and 12-representation prerequisites; audit under the shared lock')
    parser.add_argument('--idle-grace-seconds', type=float, default=180,
                        help='Exit after this long without a queue process or lock when prerequisites are missing')
    parser.add_argument('--run', action='store_true', help='Run frozen held-out evaluation after all checks')
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.idle_grace_seconds < 0:
        parser.error('poll-seconds must be 1..60 and idle-grace-seconds nonnegative')
    if bool(args.amend_from) != bool(args.alignment_audit):
        parser.error('--amend-from and --alignment-audit must be provided together')
    if not args.run:
        print('Dry-run: wait for all prerequisite queues if requested, audit dev selection, then plan/freeze, evaluate, collect and generate the handoff. No test or GPU touched.')
        return
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    lock = (work / 'finish.lock').open('a+')
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.exit(2, 'Error: another finish process owns this work directory\n')
    try:
        wait_until_ready(args.search_evidence, args.classic_evidence, args.poll_seconds,
                         args.idle_grace_seconds if args.wait_until_ready else 0)
        journal = work / 'finish_journal.jsonl'
        done = load_journal(journal)
        py = sys.executable
        plan, selection, frozen = work/'plan.json', work/'selection_evidence.json', work/'frozen.json'
        results, csv_path, f8 = work/'evaluation', work/'B.csv', work/'f8_handoff'
        figures = work/'latex_handoff'/'figures'
        run_step('plan', [py, '-B', '-m', 'models.neural.run_b_evaluation', 'plan', '--output', plan,
                          '--search-evidence', args.search_evidence], [plan], work, journal, done)
        draft = read_json(plan)
        if draft.get('partial') or draft.get('missing_requirements'):
            raise ValueError('incomplete training inventory cannot be frozen')
        evidence = selection_evidence(draft, args.classic_evidence)
        if selection.exists():
            if read_json(selection) != evidence:
                raise ValueError('existing selection evidence differs; no overwrite permitted')
        else:
            write_new_json(selection, evidence)
        classic_plan = work/'classic_plan.json'
        run_step('classic_plan', [args.classic_python, '-B', '-m', 'models.neural.classic_frozen_predict_b',
                                  'prepare', '--output', classic_plan], [classic_plan], work, journal, done)
        if args.amend_from:
            freeze_command = [py, '-B', '-m', 'models.neural.amend_alignment_b', '--parent', args.amend_from,
                              '--plan', plan, '--selection', selection, '--audit', args.alignment_audit, '--output', frozen]
        else:
            freeze_command = [py, '-B', '-m', 'models.neural.run_b_evaluation', 'freeze', '--plan', plan,
                              '--selection-evidence', selection, '--output', frozen]
        run_step('freeze', freeze_command, [frozen], work, journal, done)
        manifest = load_frozen(frozen)
        print('Primary validation model:', manifest['selection']['primary_method'], flush=True)
        run_step('evaluate', [py, '-B', '-m', 'models.neural.run_b_evaluation', 'execute', '--frozen', frozen,
                              '--output-root', results, '--device', 'mps', '--run'], [], work, journal, done, resume_output=True, recheck=True)
        missing = [task['id'] for task in manifest['tasks'] if not (results/'tasks'/task['id']/'scores.json').is_file()]
        if missing:
            raise ValueError(f'evaluation left {len(missing)} incomplete tasks, e.g. {missing[0]}')
        if args.amend_from:
            prior_root = args.parent_evaluation_root or args.amend_from.resolve().parent/'evaluation'
            comparison = work/'version_comparison'
            run_step('compare_parent', [py, '-B', '-m', 'models.neural.compare_evaluation_versions_b',
                                        '--parent-frozen', args.amend_from, '--parent-root', prior_root,
                                        '--new-frozen', frozen, '--new-root', results, '--output-dir', comparison],
                     [comparison/'summary.json', comparison/'per_task.csv'], work, journal, done)
        classic_dirs = []
        for source in ('conll2003', 'wnut17', 'mit_restaurant'):
            name = f'classic_predict_{source}'
            if name in done:
                command = done[name]['command']
                out = Path(command[command.index('--output-dir') + 1])
            else:
                out = work/'classic'/source
                attempt = 1
                while out.exists():
                    attempt += 1
                    out = work/'classic'/f'{source}_attempt_{attempt}'
            run_step(name, [args.classic_python, '-B', '-m', 'models.neural.classic_frozen_predict_b', 'predict',
                            '--classic-plan', classic_plan, '--frozen', frozen, '--source', source,
                            '--output-dir', out], [out/'metadata.json'], work, journal, done)
            classic_dirs.append(out)
        classic_scores = work/'classic_scores'
        classic_csv = classic_scores/'classic_B.csv'
        run_step('classic_score', [py, '-B', '-m', 'models.neural.score_classic_b', '--frozen', frozen,
                                   '--evaluation-root', results, '--classic-dir', *classic_dirs,
                                   '--output-dir', classic_scores], [classic_csv, classic_scores/'inventory.json'], work, journal, done)
        run_step('collect', [py, '-B', '-m', 'models.neural.collect_b_evaluations', '--frozen', frozen,
                             '--output-root', results, '--csv', csv_path], [csv_path], work, journal, done, resume_output=True)
        run_step('f8', [py, '-B', '-m', 'models.neural.handoff_f8_b', '--frozen', frozen, '--csv', csv_path,
                        '--output-dir', f8], [f8/'f8_b_by_seed.csv', f8/'f8_b_summary.csv', f8/'f8_b_draft.md'], work, journal, done)
        c_handoff = work/'c_verified'
        run_step('verify_c_t3', [py, '-B', '-m', 'models.neural.verify_c_t3_b', '--frozen', frozen,
                                  '--evaluation-root', results, '--c-csv', ROOT/'results/C.csv', '--output-dir', c_handoff],
                 [c_handoff/'C_verified.csv', c_handoff/'inventory.json'], work, journal, done)
        report = work/'latex_handoff'
        run_step('report', [py, '-B', '-m', 'models.neural.report_b', '--csv', classic_csv, csv_path,
                            '--methods', 'M0', 'M1', 'M2', 'M3', 'M4', 'Span',
                            '--c-csv', c_handoff/'C_verified.csv', '--frozen', frozen, '--output-dir', report, '--latex-prefix', 'b_handoff'],
                 [report/'t3_b_native_full.tex', report/'t3_b_300.tex', report/'t3_b_seed_summary.tex', report/'b_sections.tex'], work, journal, done)
        analysis = work/'analysis'
        gold = ROOT/'data/processed/conll2003/test.jsonl'
        classic_conll = classic_dirs[0]
        predictions = {}
        for method in ('M0', 'M1', 'M2'):
            predictions[method] = classic_conll/'tasks'/f'conll2003__{method}__full'/'predictions.jsonl'
        for method in ('M3', 'M4', 'Span'):
            matching = [task for task in manifest['tasks'] if task['group'] == 'domain'
                        and task['method'] == method and task['target'] == 'conll2003'
                        and task['scope'] == 'full' and task['seed'] == 0]
            if len(matching) != 1:
                raise ValueError(f'missing unique CoNLL domain prediction: {method}')
            predictions[method] = results/'tasks'/matching[0]['id']/'predictions.jsonl'
        paired_command = [py, '-B', '-m', 'models.neural.score_b', '--gold', gold]
        for method, path in predictions.items():
            paired_command += ['--pred', f'{method}={path}']
        paired_command += ['--entity-types', 'LOC', 'MISC', 'ORG', 'PER', '--output', analysis/'paired_models_conll_seed0.json']
        run_step('paired_models', paired_command, [analysis/'paired_models_conll_seed0.json'], work, journal, done)
        run_step('boundary_compare', [py, '-B', '-m', 'models.neural.compare_boundary_b', '--gold', gold,
                                      '--pred', f'M4={predictions["M4"]}', '--pred', f'Span={predictions["Span"]}',
                                      '--output', analysis/'boundary_m4_span_conll_seed0.json'],
                 [analysis/'boundary_m4_span_conll_seed0.json'], work, journal, done)
        extra = work/'additional_analysis'
        run_step('additional_analysis', [py, '-B', '-m', 'models.neural.additional_b_analysis',
                                         '--frozen', frozen, '--evaluation-root', results, '--output-dir', extra],
                 [extra/'inventory.json'], work, journal, done)
        plot = [str(args.plot_python), '-B', '-m', 'models.neural.plot_scores_b']
        four = 'LOC+MISC+ORG+PER'
        run_step('figure_f3b', [*plot, 'f3b', '--csv', csv_path, '--output', figures/'f3b_b.pdf', '--experiment', 'representation', '--label-space', four], [figures/'f3b_b.pdf'], work, journal, done)
        run_step('figure_f6a', [*plot, 'f6a', '--csv', csv_path, '--output', figures/'f6a_b.pdf', '--label-space', 'LOC+ORG+PER'], [figures/'f6a_b.pdf'], work, journal, done)
        run_step('figure_f6b', [str(args.plot_python), '-B', '-m', 'models.neural.plot_overlap_f6b_b',
                                 '--frozen', frozen, '--input', ROOT/'results/m2_transfer_overlap.json',
                                 '--source', 'conll2003', '--output', figures/'f6b_from_a_m2.pdf',
                                 '--metadata', analysis/'f6b_from_a_m2.json'],
                 [figures/'f6b_from_a_m2.pdf', analysis/'f6b_from_a_m2.json'], work, journal, done)
        run_step('figure_f2', [*plot, 'f2', '--csv', classic_csv, csv_path, '--output', figures/'f2_b.pdf', '--label-space', four], [figures/'f2_b.pdf'], work, journal, done)
        bert_task = next(task for task in manifest['tasks'] if task['group'] == 'domain' and task['method'] == 'M4'
                         and task['target'] == 'conll2003' and task['scope'] == 'full' and task['seed'] == 0)
        validation = ROOT/'data/processed/conll2003/validation.jsonl'
        base_vectors, fine_vectors = work/'vectors'/'base_validation.jsonl', work/'vectors'/'finetuned_validation.jsonl'
        export = [py, '-B', '-m', 'models.neural.export_bert_entities', '--input', validation, '--device', 'mps', '--offline']
        run_step('vectors_base', [*export, '--model', 'bert-base-cased', '--output', base_vectors], [base_vectors], work, journal, done, use_mps_lock=True)
        run_step('vectors_finetuned', [*export, '--model', Path(bert_task['checkpoint']).parent, '--fine-tuned', '--output', fine_vectors], [fine_vectors], work, journal, done, use_mps_lock=True)
        glove_vectors, glove_meta = work/'vectors'/'glove_validation.jsonl', work/'vectors'/'glove_validation_metadata.json'
        run_step('vectors_glove', [str(args.plot_python), '-B', '-m', 'models.neural.export_glove_entities_b',
                                   '--gold', validation, '--glove', ROOT/'data/external/glove/glove.6B.100d.txt',
                                   '--output', glove_vectors, '--metadata', glove_meta],
                 [glove_vectors, glove_meta], work, journal, done)
        run_step('vector_inventory', [str(args.plot_python), '-B', '-m', 'models.neural.vector_manifest_b',
                                      '--gold', validation, '--glove', glove_vectors, '--base-bert', base_vectors,
                                      '--fine-tuned-bert', fine_vectors, '--output', work/'vectors'/'manifest.json'],
                 [work/'vectors'/'manifest.json'], work, journal, done)
        run_step('figure_f4', [str(args.plot_python), '-B', '-m', 'models.neural.plot_entity_b', '--gold', validation,
                               '--glove', ROOT/'data/external/glove/glove.6B.100d.txt', '--base-bert', base_vectors,
                               '--fine-tuned-bert', fine_vectors, '--max-samples', '1000', '--seed', '42',
                               '--output', figures/'f4_b.pdf', '--metrics', analysis/'f4_metrics.json'],
                 [figures/'f4_b.pdf', analysis/'f4_metrics.json'], work, journal, done)
        clusters, assignments = analysis/'f4_clusters_all_validation.json', analysis/'f4_assignments_all_validation.jsonl'
        run_step('misc_clusters', [str(args.plot_python), '-B', '-m', 'analysis.cluster_entities', '--gold', validation,
                                   '--glove', ROOT/'data/external/glove/glove.6B.100d.txt', '--base-bert', base_vectors,
                                   '--fine-tuned-bert', fine_vectors, '--misc-k', '4', '--output', clusters,
                                   '--assignments', assignments], [clusters, assignments], work, journal, done)
        run_step('misc_review_material', [str(args.plot_python), '-B', '-m', 'models.neural.review_misc_clusters_b',
                                          '--gold', validation, '--metrics', clusters, '--assignments', assignments,
                                          '--phase', 'validation', '--output', analysis/'misc_review.json',
                                          '--markdown', analysis/'misc_review.md'],
                 [analysis/'misc_review.json', analysis/'misc_review.md'], work, journal, done)
        print('Finished B post-processing:', work, flush=True)
    except (OSError, ValueError, KeyError, RuntimeError, BlockingIOError) as error:
        parser.exit(2, f'Error: {error}\n')
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
