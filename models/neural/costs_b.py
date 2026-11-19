"""Generate a role-B cost handoff from frozen, already-produced metadata.

No GPU, new inference, model loading or test data are involved. Missing
profiles or unverifiable training times remain blank, never borrowed from a
different task. Outputs are create-only and include a SHA-256 input inventory.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import os


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT/'output/b_finish_v2'
RESULTS = ROOT/'results'
FIELDS = (
    'task_id','experiment','condition','method','train_source','dataset','scope','seed',
    'device','sentences','inference_batch_size','inference_seconds','sentences_per_second',
    'parameter_count','trainable_parameter_count','mps_sampled_max_current_allocated_bytes',
    'mps_sampled_max_driver_allocated_bytes','mps_batch_snapshot_attempts',
    'profile_status','prediction_reused_from','training_run',
    'training_wall_seconds','training_wall_basis','training_epoch_seconds_sum',
    'training_epochs_measured','training_epoch_basis','metadata_sha256','profile_sha256',
    'training_config_sha256','training_log_sha256','training_journal_sha256',
)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def inside(root,value):
    path=Path(value);resolved=(path if path.is_absolute() else root/path).resolve()
    if not resolved.is_relative_to(root.resolve()):raise ValueError(f'path outside project: {value}')
    return resolved


def read_frozen(path):
    obj=json.loads(path.read_text(encoding='utf-8'));signed=dict(obj);digest=signed.pop('manifest_sha256',None)
    canonical=json.dumps(signed,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
    if (obj.get('state')!='frozen' or obj.get('version')!=1 or obj.get('partial') is not False
            or obj.get('missing_requirements')!=[] or hashlib.sha256(canonical).hexdigest()!=digest):
        raise ValueError('invalid or modified frozen manifest')
    return obj


def journal_index(root,inventory):
    """Link completed queue task starts to ends in the same journal and run path."""
    result={}
    for path in sorted((root/'output/b_training_queue').glob('*.jsonl')):
        if not path.name.startswith(('run_','representations_','tuning_')):continue
        inventory[str(path)]=sha256(path)
        active={}
        for line in path.read_text(encoding='utf-8').splitlines():
            record=json.loads(line);kind=record.get('event');output=record.get('output')
            if not output:continue
            key=inside(root,output)
            if kind=='task_start':
                if key in active:raise ValueError(f'duplicate task start in {path}: {key}')
                active[key]=record
            elif kind=='task_complete' and key in active:
                start=active.pop(key)
                if key in result:raise ValueError(f'multiple successful queue intervals: {key}')
                if 'elapsed_seconds' in start and 'elapsed_seconds' in record:
                    seconds=float(record['elapsed_seconds'])-float(start['elapsed_seconds'])
                    basis='queue monotonic elapsed seconds'
                else:
                    seconds=(datetime.fromisoformat(record['timestamp'])-datetime.fromisoformat(start['timestamp'])).total_seconds()
                    basis='queue wall-clock timestamps, 1 s resolution'
                if not math.isfinite(seconds) or seconds<0:raise ValueError(f'invalid queue interval: {key}')
                result[key]={'seconds':seconds,'basis':basis,'journal':path,'log':inside(root,start['log'])}
    return result


def epoch_times(log,epochs):
    """Only a complete ordered run with all epoch timers and Done is usable."""
    text=log.read_text(encoding='utf-8',errors='replace')
    patterns=(r'^Epoch\s+(\d+)\s+(?:finished|completed)\s+in\s+([0-9]+(?:\.[0-9]+)?)s',
              r'^Epoch\s+(\d+)\s*\|\s*seconds=([0-9]+(?:\.[0-9]+)?)')
    matches=[]
    for pattern in patterns:
        matches.extend(re.finditer(pattern,text,re.MULTILINE))
    matches.sort(key=lambda match:match.start())
    if [int(match.group(1)) for match in matches]!=list(range(1,epochs+1)):
        return None
    done=list(re.finditer(r'^Done\.(?:\s|$)',text,re.MULTILINE))
    if not done or done[-1].start()<matches[-1].start():return None
    values=[float(match.group(2)) for match in matches]
    if not all(math.isfinite(x) and x>=0 for x in values):return None
    return sum(values),len(values)


def training_details(root,run_dir,queue,inventory,cache):
    if run_dir in cache:return cache[run_dir]
    config_path=run_dir/'run_config.json';config=json.loads(config_path.read_text(encoding='utf-8'))
    config_sha=sha256(config_path);inventory[str(config_path)]=config_sha
    epochs=config.get('epochs');data={'training_config_sha256':config_sha}
    if type(epochs) is not int or epochs<1:raise ValueError(f'invalid training epochs: {run_dir}')
    event=queue.get(run_dir);log=event['log'] if event else run_dir.with_suffix('.log')
    if event:
        data.update(training_wall_seconds=event['seconds'],training_wall_basis=event['basis'],
                    training_journal_sha256=inventory[str(event['journal'])])
    if log.is_file():
        log_sha=sha256(log);inventory[str(log)]=log_sha;data['training_log_sha256']=log_sha
        timers=epoch_times(log,epochs)
        if timers:
            data.update(training_epoch_seconds_sum=timers[0],training_epochs_measured=timers[1],
                        training_epoch_basis='sum of all logged epoch timers; not total training wall time')
    cache[run_dir]=data
    return data


def collect(root,work):
    root,work=root.resolve(),work.resolve();manifest_path=work/'frozen.json';manifest=read_frozen(manifest_path)
    tasks=manifest.get('tasks')
    if not isinstance(tasks,list) or len(tasks)!=162:raise ValueError('expected exactly 162 frozen tasks')
    inventory={str(manifest_path):sha256(manifest_path)}
    for relative in ('models/neural/costs_b.py','models/neural/predict_neural.py'):
        source=root/relative
        inventory[str(source)]=sha256(source)
        if relative=='models/neural/predict_neural.py' and inventory[str(source)]!=manifest['code_sha256'][relative]:
            raise ValueError('inference profiler code differs from frozen manifest')
    queue=journal_index(root,inventory);cache={};rows=[];seen=set()
    for task in tasks:
        task_id=task['id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+',task_id) or task_id in seen:raise ValueError(f'invalid task ID: {task_id}')
        seen.add(task_id);directory=work/'evaluation/tasks'/task_id;meta_path=directory/'metadata.json'
        meta=json.loads(meta_path.read_text(encoding='utf-8'));meta_sha=sha256(meta_path);inventory[str(meta_path)]=meta_sha
        if meta.get('task')!=task or meta.get('frozen_manifest_sha256')!=manifest['manifest_sha256']:
            raise ValueError(f'{task_id}: metadata not bound to frozen task')
        run_dir=inside(root,task['run_dir']);run=manifest['runs'][str(run_dir)]
        if meta.get('checkpoint_sha256')!=run['checkpoint_sha256'] or meta.get('device')!='mps':
            raise ValueError(f'{task_id}: checkpoint/device provenance mismatch')
        row={'task_id':task_id,'experiment':task['experiment'],'condition':task['condition'],
             'method':task['method'],'train_source':task['train_source'],'dataset':task['target'],
             'scope':task['scope'],'seed':task['seed'],'device':meta['device'],'sentences':meta['sentences'],
             'prediction_reused_from':meta.get('prediction_reused_from') or '',
             'training_run':str(run_dir.relative_to(root)),'metadata_sha256':meta_sha}
        profile=directory/'prediction_profile.json';expected=meta.get('profile_sha256')
        if expected is None:
            if profile.exists():raise ValueError(f'{task_id}: unexpected unbound profile')
            row['profile_status']='reused_prediction_no_profile' if meta.get('prediction_reused_from') else 'not_recorded'
        else:
            if not profile.is_file() or sha256(profile)!=expected:raise ValueError(f'{task_id}: changed profile')
            inventory[str(profile)]=expected;obj=json.loads(profile.read_text(encoding='utf-8'));memory=obj['memory_sampling']
            if obj['device']!='mps' or obj['sentences']!=meta['sentences'] or memory.get('is_exact_peak') is not False:
                raise ValueError(f'{task_id}: incompatible profile')
            seconds=float(obj['seconds']);speed=float(obj['sentences_per_second'])
            if not math.isfinite(seconds) or seconds<=0 or not math.isclose(speed,obj['sentences']/seconds,rel_tol=1e-9):
                raise ValueError(f'{task_id}: invalid timing values')
            row.update(inference_batch_size=obj['batch_size'],inference_seconds=seconds,sentences_per_second=speed,
                       parameter_count=obj['parameter_count'],trainable_parameter_count=obj['trainable_parameter_count'],
                       mps_sampled_max_current_allocated_bytes=memory.get('sampled_max_current_allocated_bytes'),
                       mps_sampled_max_driver_allocated_bytes=memory.get('sampled_max_driver_allocated_bytes'),
                       mps_batch_snapshot_attempts=memory.get('batch_sample_attempts'),profile_status='measured',profile_sha256=expected)
        row.update(training_details(root,run_dir,queue,inventory,cache));rows.append(row)
    return rows,inventory,manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',help='Create the three handoff files after validation')
    args=parser.parse_args()
    try:
        destinations=[RESULTS/'B_costs.csv',RESULTS/'B_costs_README.md',RESULTS/'B_costs_manifest.json']
        if any(p.exists() or p.is_symlink() for p in destinations):raise FileExistsError('one or more B_costs output files already exist')
        rows,inventory,manifest=collect(ROOT,WORK)
        counts={status:sum(row['profile_status']==status for row in rows) for status in ('measured','reused_prediction_no_profile','not_recorded')}
        print(f'Verified {len(rows)} frozen tasks; profile status: {counts}')
        if not args.run:
            print('Dry-run only; use --run to create the CSV, README and input SHA inventory.')
            return 0
        readme='''# Role B model cost handoff\n\n'''
        readme+=f"Frozen manifest: `{manifest['manifest_sha256']}`. CSV has {len(rows)} rows, one per frozen test task. `B_costs_manifest.json` records every input SHA-256 and output SHA-256.\n\n"
        readme+='''`inference_seconds` times the synchronized predictor call after model/tokenizer loading and after reading prepared token rows. It includes subword tokenization/encoding of prepared words, model forward, decoding, and two synchronized memory snapshots per batch. It excludes dataset preparation, external raw-text/spaCy tokenization, process startup, model loading, and writing predictions. Divide the recorded sentence count by these seconds for throughput. Batch size, length distribution, model, dataset and profiling synchronization differ; these are not uniform end-to-end or production throughput measurements. Parameter counts are taken from the restored inference model; `trainable_parameter_count` counts parameters marked `requires_grad` at inference, which is not independently a record of the optimizer's training parameter groups.\n\nThe two MPS allocation columns are maxima of synchronized *batch snapshots*, not exact peak memory. Live tensor allocation and process-level Metal driver allocation measure different things; driver values include cached pools. Apple Silicon has shared unified memory, so neither is dedicated VRAM nor total physical memory. The profiler can miss transient intra-batch maxima. Blank profile fields mean no measurement: a fixed-300 task may reuse a full-set prediction without a fresh profile, while the four representation variants did not generate profiles. Do not borrow the full-set throughput for a subset.\n\nTraining wall time is recorded only when the same queue journal contains a matching task_start/task_complete for the exact run. Queue monotonic intervals and one-second-resolution wall-clock intervals are identified separately; both can include startup, loading, epoch work, validation, checkpoint writes and scheduling delays. `training_epoch_seconds_sum` is a separate sum of all explicitly logged epoch timers, available only if each requested epoch and final Done are recorded; it is not total wall time and timing boundaries vary by training script. No cost is inferred for missing history. The same trained model appears in many evaluation tasks, so training values repeat and must not be summed across CSV rows. Historical MPS true peak memory was not recorded.\n'''
        with tempfile.TemporaryDirectory(prefix='.b_costs_',dir=RESULTS) as temp:
            staging=Path(temp);csvpath=staging/'B_costs.csv'
            with csvpath.open('x',encoding='utf-8',newline='') as dest:
                writer=csv.DictWriter(dest,fieldnames=FIELDS,extrasaction='raise');writer.writeheader();writer.writerows(rows)
            notes=staging/'B_costs_README.md';notes.write_text(readme,encoding='utf-8')
            payload={'frozen_manifest_sha256':manifest['manifest_sha256'],'input_sha256':inventory,
                     'output_sha256':{'results/B_costs.csv':sha256(csvpath),'results/B_costs_README.md':sha256(notes)},
                     'tasks':len(rows),'profile_status_counts':counts}
            check=staging/'B_costs_manifest.json';check.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            for source,dest in zip((csvpath,notes,check),destinations,strict=True):os.link(source,dest)
        print('Created results/B_costs.csv, B_costs_README.md and B_costs_manifest.json')
        return 0
    except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__=='__main__':raise SystemExit(main())
