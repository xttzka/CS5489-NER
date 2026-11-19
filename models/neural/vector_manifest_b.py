"""Validate and inventory A-compatible gold-entity vector handoff files."""
import argparse
import hashlib
import json
from pathlib import Path
from analysis.cluster_entities import read_bert_vectors, read_gold


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('gold','glove','base-bert','fine-tuned-bert','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    try:
        if a.output.exists() or a.output.resolve() in {x.resolve() for x in (a.gold,a.glove,a.base_bert,a.fine_tuned_bert)}:
            raise ValueError('output must be new and distinct from all inputs')
        gold=read_gold(a.gold); total=len(gold); vectors={}; reports={}
        for name,path in (('glove',a.glove),('base_bert',a.base_bert),('fine_tuned_bert',a.fine_tuned_bert)):
            values,meta=read_bert_vectors(path,gold); vectors[name]=set(values)
            reports[name]={**meta,'entities':len(values),'missing_entities':total-len(values),
                           'missing_fraction':(total-len(values))/total,'sha256':digest(path)}
        common=set.intersection(*vectors.values())
        manifest={'gold':str(a.gold),'gold_sha256':digest(a.gold),'total_gold_entities':total,
                  'representations':reports,'common_entities':len(common),
                  'common_fraction':len(common)/total,'key':'sample_id,start,end (end exclusive)',
                  'note':'Entity vectors are derived from the explicitly named input split; no held-out results are inferred.'}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with a.output.open('x',encoding='utf-8') as out:
            json.dump(manifest,out,ensure_ascii=False,indent=2);out.write('\n')
        print(f'Verified {len(common)}/{total} matched entities across all three vector files')
    except (OSError,ValueError,KeyError,TypeError) as error:
        p.exit(2,f'Error: {error}\n')


if __name__=='__main__': main()
