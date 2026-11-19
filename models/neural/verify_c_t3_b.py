"""After freeze, verify C's existing fixed-300 predictions before T3 handoff.

No models are run. Missing or mismatched C artifacts become TODOs rather than
numbers. Each accepted point must match the frozen B input/subset hashes, exact
IDs, native label schema, declared model/prompt configuration, original report,
C.csv row and an independent score recomputation.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from eval.result_tables import FIELDS, read_csv
from eval.score_ner import read_subset_ids, read_tags_by_id, score_ner
from models.llm.common import EXAMPLE_SOURCE, TYPE_DESCRIPTIONS
from models.neural.run_b_evaluation import ROOT, load_frozen, sha256

TARGETS=('conll2003','conllpp','wikigold','btc','wnut17','mit_restaurant')
METHODS={'GLiNER':('gliner',None,0), 'Qwen1.5B-zero':('qwen1.5b_k0_plain','Qwen/Qwen2.5-1.5B-Instruct',0),
         'Qwen7B-zero':('qwen7b_k0_plain','Qwen/Qwen2.5-7B-Instruct',0),
         'Qwen7B-few':('qwen7b_k10_similar_plain','Qwen/Qwen2.5-7B-Instruct',10)}


def run(args):
    frozen=load_frozen(args.frozen)
    old=list(read_csv(args.c_csv)) if args.c_csv.is_file() else []
    output=[]; missing=[]
    for method,(prefix,model,shots) in METHODS.items():
        for target in TARGETS:
            try:
                name=(f'gliner_{target}_test_test_300.json' if method=='GLiNER'
                      else f'{prefix}_test_300_{target}_test_seed0.json')
                report_path=ROOT/'results/c'/name; report=json.loads(report_path.read_text(encoding='utf-8'))
                if report.get('dataset')!=target or report.get('split')!='test' or report.get('scope')!='test_300' or report.get('input'):
                    raise ValueError('report is not the required target test subset')
                if report.get('label_scheme',target)!=target:
                    raise ValueError('prediction label scheme differs from target native labels')
                if method=='GLiNER':
                    if report.get('method')!='GLiNER' or report.get('model')!='urchade/gliner_medium-v2.1' or report.get('threshold')!=0.5: raise ValueError('GLiNER configuration differs')
                elif (report.get('model')!=model or report.get('shots')!=shots or report.get('guideline') is not False
                      or report.get('seed')!=0 or report.get('max_new_tokens')!=256
                      or (shots and (report.get('selection')!='similar' or report.get('example_source')!=f'{EXAMPLE_SOURCE.get(target,target)}:train'))
                      or report.get('adapter') or report.get('wording') or report.get('example_pool_size')):
                    raise ValueError('LLM model or prompting configuration differs')
                subset=ROOT/'data/subsets'/target/'test_300.txt'; gold_path=ROOT/'data/processed'/target/'test.jsonl'
                if (ROOT/str(report.get('subset_ids',''))).resolve()!=subset.resolve(): raise ValueError('report subset path differs')
                for path in (subset,gold_path):
                    key=hashlib.sha256(str(path.resolve()).encode()).hexdigest()+'.json'
                    binding=json.loads((args.evaluation_root/'input_hashes'/key).read_text(encoding='utf-8'))
                    if binding.get('sha256')!=sha256(path) or binding.get('frozen_manifest_sha256')!=frozen['manifest_sha256']:
                        raise ValueError('gold/subset differs from B evaluation binding')
                ids=read_subset_ids(subset)
                if len(ids)!=300 or len(set(ids))!=300:raise ValueError('subset must have 300 distinct IDs')
                gold_all=read_tags_by_id(gold_path)
                if set(ids)-gold_all.keys():raise ValueError('subset includes missing gold IDs')
                gold={key:gold_all[key] for key in ids}
                pred_path=(ROOT/report['predictions']).resolve()
                if not pred_path.is_relative_to(ROOT.resolve()):raise ValueError('prediction path outside project')
                predictions=read_tags_by_id(pred_path)
                if set(predictions)!=set(ids):raise ValueError('prediction IDs differ from fixed subset')
                score=score_ner(gold,predictions)
                for key in ('tp','fp','fn','precision','recall','f1'):
                    if abs(float(score[key])-float(report['scores'][key]))>1e-10:raise ValueError(f'{key} disagrees with report')
                ref=f'results/c/{name}#/scores/f1'
                matched=[row for row in old if row['source_ref']==ref and row['dataset']==target and row['scope']=='test_300'
                         and row['split']=='test' and row['metric']=='f1' and row['entity_type']=='all'
                         and row['label_space']=='+'.join(sorted(TYPE_DESCRIPTIONS[target]))]
                if len(matched)!=1 or abs(float(matched[0]['value'])-score['f1'])>1e-10:
                    raise ValueError('C.csv does not uniquely match the independently verified report')
                row={key:'' for key in FIELDS}
                row.update(member='C',experiment='baseline',method=method,condition='standard',
                           train_source=(report.get('example_source') or '').split(':')[0],dataset=target,
                           split='test',scope='test_300',label_space='+'.join(sorted(TYPE_DESCRIPTIONS[target])),
                           entity_type='all',entity_group='all',seed=0,metric='f1',value=score['f1'],unit='ratio',
                           config_json=json.dumps({'model':report['model'],'shots':shots,'frozen_manifest_sha256':frozen['manifest_sha256'],
                                                   'prediction_sha256':sha256(pred_path),'subset_sha256':sha256(subset),
                                                   'report_sha256':sha256(report_path),'c_csv_sha256':sha256(args.c_csv)},sort_keys=True),
                           source_ref=ref)
                output.append(row)
            except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as error:
                missing.append({'method':method,'dataset':target,'reason':str(error)})
    return output,missing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,required=True);p.add_argument('--evaluation-root',type=Path,required=True)
    p.add_argument('--c-csv',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.output_dir.exists():raise FileExistsError(f'result exists: {a.output_dir}')
        rows,missing=run(a);a.output_dir.parent.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(prefix='.c_t3_',dir=a.output_dir.parent) as td:
            staged=Path(td)/'result';staged.mkdir()
            with (staged/'C_verified.csv').open('x',encoding='utf-8',newline='') as out:
                writer=csv.DictWriter(out,fieldnames=FIELDS,lineterminator='\n');writer.writeheader();writer.writerows(rows)
            (staged/'inventory.json').write_text(json.dumps({'verified':len(rows),'missing':missing},indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
            staged.rename(a.output_dir)
        print(f'C T3 points verified={len(rows)}; TODO={len(missing)}; {a.output_dir}')
    except (OSError,ValueError,KeyError,TypeError) as error:p.exit(2,f'Error: {error}\n')


if __name__=='__main__':main()
