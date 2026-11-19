"""Post-freeze B familiarity and paired intervention analyses.

Requires complete, validated task predictions. Reads held-out gold only after
validating the frozen manifest. Writes a new directory; no GPU is used.
"""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from eval.baseline_statistics import bootstrap_counts, metric_values
from eval.score_familiarity import FAMILIARITY, news_settings, score_full_test, training_names
from eval.score_ner import read_tags_by_id, score_ner
from models.neural.run_b_evaluation import ROOT, load_frozen, sha256, verify_result


def task_one(manifest, **attrs):
    matches = [task for task in manifest['tasks'] if all(task.get(k)==v for k,v in attrs.items())]
    if len(matches)!=1:
        raise ValueError(f'expected one frozen task for {attrs}; found {len(matches)}')
    return matches[0]


def checked_prediction(manifest, root, task):
    folder=root/'tasks'/task['id']
    meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    run=manifest['runs'][str(Path(task['run_dir']).resolve())]
    verify_result(folder,task,manifest['manifest_sha256'],meta['input_sha256'],meta['subset_sha256'],
                  run['checkpoint_sha256'],manifest['bootstrap']['resamples'],manifest['bootstrap']['seed'])
    path=ROOT/task['input']
    if sha256(path)!=meta['input_sha256']:
        raise ValueError(f'input changed: {path}')
    return folder/'predictions.jsonl', path


def paired(gold_a,pred_a,gold_b,pred_b):
    sets=[set(x) for x in (gold_a,pred_a,gold_b,pred_b)]
    if not sets[0] or any(x!=sets[0] for x in sets[1:]):
        raise ValueError('paired intervention IDs are not identical')
    counts=np.zeros((len(sets[0]),2,1,3),dtype=np.int64)
    for index,key in enumerate(sorted(sets[0])):
        for column,(gold,pred) in enumerate(((gold_a,pred_a),(gold_b,pred_b))):
            record=score_ner({key:gold[key]},{key:pred[key]})
            counts[index,column,0]=[record['tp'],record['fp'],record['fn']]
    point=metric_values(counts.sum(axis=0))
    boot=metric_values(bootstrap_counts(counts,1000,42))
    diff=boot[:,1,0,2]-boot[:,0,0,2]
    return {'sentence_count':len(sets[0]),'first_f1':float(point[0,0,2]),'second_f1':float(point[1,0,2]),
            'second_minus_first':float(point[1,0,2]-point[0,0,2]),'paired_ci95':np.quantile(diff,(.025,.975)).tolist(),
            'bootstrap':{'unit':'sentence','resamples':1000,'seed':42,'interval':'95% percentile'}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--evaluation-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    try:
        manifest=load_frozen(args.frozen)
        if args.output_dir.exists():
            raise FileExistsError(f'refusing overwrite: {args.output_dir}')
        if manifest['bootstrap'] != {'resamples':1000,'seed':42}:
            raise ValueError('bootstrap protocol differs')
        types=news_settings()['types']; label_map={kind:kind for kind in types}
        names=training_names()
        args.output_dir.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.additional_b_',dir=args.output_dir.parent) as tmp:
            staged=Path(tmp)/'result'; staged.mkdir()
            for method in ('M3','M4','Span'):
                for target,group in (('conll2003','domain'),('conllpp','temporal')):
                    task=task_one(manifest,group=group,method=method,target=target,scope='full',seed=0)
                    pred,_=checked_prediction(manifest,args.evaluation_root,task)
                    scores=score_full_test(target,pred,names,label_map)
                    result={'dataset':target,'method':method,'seed':0,'label_space':types,
                            'familiarity':FAMILIARITY,'frozen_manifest_sha256':manifest['manifest_sha256'],
                            'prediction_sha256':sha256(pred),
                            'training_reference_sha256':sha256(ROOT/'data/processed/conll2003/train.jsonl'),
                            'entity_index_sha256':sha256(ROOT/'data/derived/entity_index'/target/'test_v1.jsonl'),**scores}
                    (staged/f'familiarity__{method}__{target}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
                base=task_one(manifest,group='domain',method=method,target='conll2003',scope='full',seed=0)
                base_pred,base_gold=checked_prediction(manifest,args.evaluation_root,base)
                for condition in ('lowercase_v1','spelling_noise_v1','no_punctuation_v1','replacement_replaced_v2'):
                    second=task_one(manifest,group='robustness',method=method,target='conll2003',condition=condition,seed=0)
                    second_pred,second_gold=checked_prediction(manifest,args.evaluation_root,second)
                    if condition=='replacement_replaced_v2':
                        first=task_one(manifest,group='robustness',method=method,target='conll2003',condition='replacement_original_v2',seed=0)
                        first_pred,first_gold=checked_prediction(manifest,args.evaluation_root,first)
                    else:
                        first_pred,first_gold=base_pred,base_gold
                    value=paired(read_tags_by_id(first_gold),read_tags_by_id(first_pred),
                                 read_tags_by_id(second_gold),read_tags_by_id(second_pred))
                    value.update(method=method,seed=0,condition=condition,first_gold=str(first_gold),second_gold=str(second_gold),
                                 first_prediction=str(first_pred),second_prediction=str(second_pred),
                                 frozen_manifest_sha256=manifest['manifest_sha256'])
                    (staged/f'paired__{method}__{condition}.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
            (staged/'inventory.json').write_text(json.dumps({'familiarity':6,'paired_interventions':12,'frozen_manifest_sha256':manifest['manifest_sha256']},indent=2)+'\n',encoding='utf-8')
            staged.rename(args.output_dir)
        print(f'Created B familiarity and paired intervention analysis: {args.output_dir}')
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__=='__main__':
    main()
