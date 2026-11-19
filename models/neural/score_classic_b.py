"""Score frozen classic predictions for F2/T3 with the B evaluation protocol.

Only run after the B manifest is frozen. Outputs are assembled in a temporary
folder and published atomically; no existing result is overwritten.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from eval.score_ner import read_subset_ids, read_tags_by_id
from eval.score_transfer import MAPPING_PATH, project_tags
from models.llm.common import TYPE_DESCRIPTIONS
from models.neural.run_b_evaluation import ROOT, load_frozen, sha256
from models.neural.score_b import append_csv, output_rows, score_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--classic-dir', type=Path, nargs=3, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        frozen = load_frozen(args.frozen)
        if args.output_dir.exists():
            raise FileExistsError(f'result exists: {args.output_dir}')
        dirs, tops = {}, {}
        for directory in args.classic_dir:
            top = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            if top.get('status') != 'complete' or top.get('frozen_manifest_sha256') != frozen['manifest_sha256']:
                raise ValueError(f'classic predictions incomplete or incompatible: {directory}')
            source = top['source']
            if source in dirs:
                raise ValueError(f'duplicate classic source: {source}')
            dirs[source] = directory
            tops[source] = top
        if set(dirs) != {'conll2003','wnut17','mit_restaurant'}:
            raise ValueError('need exactly all three classic source directories')
        mapping = json.loads(MAPPING_PATH.read_text(encoding='utf-8'))['transfer_three_types']['maps']
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.classic_scores_', dir=args.output_dir.parent) as tmp:
            staged = Path(tmp)/'result'; staged.mkdir(); rows=[]; count=0
            for source, directory in sorted(dirs.items()):
                targets = ('conll2003','conllpp','wikigold','btc') if source == 'conll2003' else (source,)
                for target in targets:
                    for scope in (('full','test_300') if target == source else ('test_300',)):
                        gold_path = ROOT/'data/processed'/target/'test.jsonl'
                        subset = ROOT/'data/subsets'/target/'test_300.txt'
                        gold = read_tags_by_id(gold_path)
                        ids = read_subset_ids(subset)
                        if len(ids) != 300 or len(set(ids)) != 300 or set(ids) - gold.keys():
                            raise ValueError(f'invalid fixed subset: {target}')
                        if scope == 'test_300':
                            gold = {sample_id:gold[sample_id] for sample_id in ids}
                        binding_path = args.evaluation_root/'input_hashes'/(hashlib.sha256(str(subset.resolve()).encode()).hexdigest()+'.json')
                        binding = json.loads(binding_path.read_text(encoding='utf-8'))
                        if binding['sha256'] != sha256(subset) or binding['frozen_manifest_sha256'] != frozen['manifest_sha256']:
                            raise ValueError(f'B and classic subset differ for {target}')
                        gold_binding_path = args.evaluation_root/'input_hashes'/(hashlib.sha256(str(gold_path.resolve()).encode()).hexdigest()+'.json')
                        gold_binding = json.loads(gold_binding_path.read_text(encoding='utf-8'))
                        if gold_binding['sha256'] != sha256(gold_path) or gold_binding['frozen_manifest_sha256'] != frozen['manifest_sha256']:
                            raise ValueError(f'B and classic gold input differ for {target}')
                        for method in ('M0','M1','M2'):
                            task_id = f'{target}__{method}__{scope}'
                            task_dir = directory/'tasks'/task_id
                            pred_path = task_dir/'predictions.jsonl'
                            meta = json.loads((task_dir/'metadata.json').read_text(encoding='utf-8'))
                            expected={'task_id':task_id,'method':method,'train_source':source,'target':target,
                                      'split':'test','scope':scope,'frozen_manifest_sha256':frozen['manifest_sha256']}
                            if any(meta.get(k)!=v for k,v in expected.items()) or meta['prediction_sha256'] != sha256(pred_path):
                                raise ValueError(f'classic prediction provenance mismatch: {task_dir}')
                            model_file = directory / {'M0':'m0_dictionary.joblib','M1':'m1_model.joblib','M2':'m2_model.crfsuite'}[method]
                            if meta.get('checkpoint_sha256') != tops[source]['model_sha256'][method] or sha256(model_file) != meta['checkpoint_sha256']:
                                raise ValueError(f'classic model checksum differs: {task_dir}')
                            if meta['subset_sha256'] != sha256(subset):
                                raise ValueError(f'classic subset changed: {task_dir}')
                            rel = gold_path.relative_to(ROOT).as_posix()
                            if meta['inputs_sha256'].get(rel) != sha256(gold_path):
                                raise ValueError(f'classic gold input changed: {gold_path}')
                            predictions=read_tags_by_id(pred_path)
                            if set(predictions) != set(gold):
                                raise ValueError(f'classic prediction IDs disagree with gold: {task_dir}')
                            effective_gold=gold
                            types=sorted(TYPE_DESCRIPTIONS[target]) if target != 'btc' else ['LOC','ORG','PER']
                            if target == 'btc':
                                effective_gold=project_tags(gold,mapping['btc'])
                                predictions=project_tags(predictions,mapping['conll2003'])
                            result=score_models(effective_gold,{method:predictions},types=types,resamples=1000,seed=42)
                            name=f'{source}__{task_id}.json'
                            with (staged/name).open('x',encoding='utf-8') as out:
                                json.dump(result,out,ensure_ascii=False,indent=2); out.write('\n')
                            context=SimpleNamespace(experiment='baseline',condition='standard',train_source=source,
                                                    dataset=target,split='test',scope=scope,model_seed=0,
                                                    scoring='transfer_three_types' if target=='btc' else 'native')
                            rows.extend(output_rows(result,context,args.output_dir/name)); count+=1
            append_csv(staged/'classic_B.csv',rows)
            (staged/'inventory.json').write_text(json.dumps({'tasks':count,'frozen_manifest_sha256':frozen['manifest_sha256'],
                                                              'bootstrap':{'resamples':1000,'seed':42}},indent=2)+'\n',encoding='utf-8')
            staged.rename(args.output_dir)
        print(f'Scored {count} classic tasks: {args.output_dir}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__ == '__main__':
    main()
