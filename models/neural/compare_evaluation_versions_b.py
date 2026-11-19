"""Compare all completed parent predictions with amended full reruns by ID.

Never selects a version using test metrics. No gold labels are read. Old and
new result roots are read-only; output is written once to a new directory.
"""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from eval.score_ner import read_tags_by_id
from models.neural.run_b_evaluation import load_frozen, read_json, sha256, verify_result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('parent-frozen','parent-root','new-frozen','new-root','output-dir'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    try:
        parent=load_frozen(a.parent_frozen);new=load_frozen(a.new_frozen)
        if new.get('amendment',{}).get('parent_manifest_sha256')!=parent['manifest_sha256'] or parent['tasks']!=new['tasks']:
            raise ValueError('manifests are not the same tasks with an explicit parent amendment')
        if a.output_dir.exists(): raise FileExistsError(f'comparison output exists: {a.output_dir}')
        rows=[]; completed_parent=0
        for task in parent['tasks']:
            old_dir=a.parent_root/'tasks'/task['id']; new_dir=a.new_root/'tasks'/task['id']
            if not old_dir.exists(): continue
            completed_parent+=1
            if not new_dir.exists():raise ValueError(f'new evaluation has not completed parent task: {task["id"]}')
            old_meta=read_json(old_dir/'metadata.json');new_meta=read_json(new_dir/'metadata.json')
            old_run=parent['runs'][str(Path(task['run_dir']).resolve())]
            new_run=new['runs'][str(Path(task['run_dir']).resolve())]
            for folder,meta,manifest,run in ((old_dir,old_meta,parent,old_run),(new_dir,new_meta,new,new_run)):
                verify_result(folder,task,manifest['manifest_sha256'],meta['input_sha256'],meta['subset_sha256'],
                              run['checkpoint_sha256'],manifest['bootstrap']['resamples'],manifest['bootstrap']['seed'])
            if old_meta['input_sha256']!=new_meta['input_sha256'] or old_meta['subset_sha256']!=new_meta['subset_sha256'] or old_run['checkpoint_sha256']!=new_run['checkpoint_sha256']:
                raise ValueError(f'inputs or checkpoint differ across versions: {task["id"]}')
            old=read_tags_by_id(old_dir/'predictions.jsonl');current=read_tags_by_id(new_dir/'predictions.jsonl')
            if set(old)!=set(current):raise ValueError(f'prediction IDs differ: {task["id"]}')
            changed_samples=changed_tokens=total_tokens=0
            for key in old:
                before,after=old[key],current[key]
                if len(before)!=len(after):raise ValueError(f'tag sequence length changed: {task["id"]}/{key}')
                differences=sum(x!=y for x,y in zip(before,after,strict=True))
                changed_samples+=bool(differences);changed_tokens+=differences;total_tokens+=len(before)
            rows.append({'task_id':task['id'],'sample_count':len(old),'changed_samples':changed_samples,
                         'token_count':total_tokens,'changed_tokens':changed_tokens,
                         'identical_file_bytes':sha256(old_dir/'predictions.jsonl')==sha256(new_dir/'predictions.jsonl'),
                         'parent_prediction_sha256':sha256(old_dir/'predictions.jsonl'),
                         'new_prediction_sha256':sha256(new_dir/'predictions.jsonl')})
        a.output_dir.parent.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(prefix='.compare_versions_',dir=a.output_dir.parent) as tmp:
            stage=Path(tmp)/'result';stage.mkdir()
            fields=('task_id','sample_count','changed_samples','token_count','changed_tokens','identical_file_bytes','parent_prediction_sha256','new_prediction_sha256')
            with (stage/'per_task.csv').open('x',encoding='utf-8',newline='') as out:
                w=csv.DictWriter(out,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
            summary={'parent_manifest_sha256':parent['manifest_sha256'],'new_manifest_sha256':new['manifest_sha256'],
                     'completed_parent_tasks_compared':completed_parent,'tasks_with_changed_predictions':sum(r['changed_samples']>0 for r in rows),
                     'changed_sentence_occurrences':sum(r['changed_samples'] for r in rows),'changed_token_occurrences':sum(r['changed_tokens'] for r in rows),
                     'note':'Counts are summed across tasks, so the same sentence can occur more than once. No test scores were used to select a version.'}
            (stage/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
            stage.rename(a.output_dir)
        print(f'Compared {completed_parent} parent tasks; changed prediction tasks={summary["tasks_with_changed_predictions"]}')
    except (OSError,ValueError,KeyError,TypeError) as error:p.exit(2,f'Error: {error}\n')


if __name__=='__main__':main()
