"""Persist A-compatible GloVe entity vectors with coverage and source hashes."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis.cluster_entities import glove_entity_vectors, read_gold


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold',type=Path,required=True)
    parser.add_argument('--glove',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    args=parser.parse_args()
    try:
        paths=[p.resolve() for p in (args.gold,args.glove,args.output,args.metadata)]
        if len(set(paths))!=len(paths) or args.output.exists() or args.metadata.exists():
            raise ValueError('input/output paths must be distinct, and outputs must not exist')
        gold=read_gold(args.gold)
        vectors,provenance=glove_entity_vectors(args.glove,gold)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.metadata.parent.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as tmp:
            staged=Path(tmp)/'vectors.jsonl'
            with staged.open('x',encoding='utf-8') as file:
                for sample_id,start,end in sorted(vectors):
                    record={'sample_id':sample_id,'start':start,'end':end,
                            'entity_type':gold[sample_id,start,end][0],
                            'vector':vectors[sample_id,start,end].tolist()}
                    file.write(json.dumps(record,ensure_ascii=False)+'\n')
            n=len(gold); present=len(vectors); partial=provenance['partially_out_of_vocabulary_spans']
            info={'gold':str(args.gold),'gold_sha256':digest(args.gold),'glove':provenance,
                  'total_gold_entities':n,'exported_entities':present,'fully_missing_entities':n-present,
                  'fully_missing_fraction':(n-present)/n,'partially_covered_entities':partial,
                  'partially_covered_fraction':partial/n,'vector_sha256':digest(staged)}
            with args.output.open('xb') as out, staged.open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''): out.write(block)
            with args.metadata.open('x',encoding='utf-8') as out:
                json.dump(info,out,ensure_ascii=False,indent=2);out.write('\n')
        print(f'Exported {present}/{n} gold entities; fully missing={n-present}; partial={partial}')
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__=='__main__': main()
