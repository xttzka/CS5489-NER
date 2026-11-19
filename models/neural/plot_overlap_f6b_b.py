"""Draw F6(b) from A's existing M2 overlap analysis after protocol freeze.

The input points originate from A's transfer analysis; this script does not
recompute data or claim that they are BERT results. OLS is descriptive.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from analysis import style


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def frozen_manifest(path):
    value=json.loads(path.read_text(encoding='utf-8'))
    saved=value.pop('manifest_sha256',None)
    actual=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if value.get('state')!='frozen' or value.get('partial') or saved!=actual:
        raise ValueError('a valid complete frozen manifest is required')
    return saved


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True,help='A results/m2_transfer_overlap.json')
    parser.add_argument('--source',default='conll2003',choices=('conll2003','wnut17','merged'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    args=parser.parse_args()
    try:
        frozen=frozen_manifest(args.frozen)
        if args.output.suffix.lower()!='.pdf' or args.output.exists() or args.metadata.exists():
            raise ValueError('output must be a new PDF and metadata must be new')
        report=json.loads(args.input.read_text(encoding='utf-8'))
        required='Fraction of target gold mentions whose full case-folded name and mapped type occur in source training.'
        if report.get('overlap_definition')!=required:
            raise ValueError('unexpected overlap definition; cannot silently reinterpret A data')
        rows=[row for row in report['points'] if row['source']==args.source and not row['target_is_source_domain']]
        if len(rows)<3 or any(row['entity_type'] not in ('PER','LOC','ORG') or not 0<=float(row['overlap'])<=1 for row in rows):
            raise ValueError('incomplete or invalid common-three-type transfer points')
        keys=[(r['target'],r['entity_type']) for r in rows]
        if len(set(keys))!=len(keys): raise ValueError('duplicate source/target/entity-type points')
        x=np.asarray([100*float(r['overlap']) for r in rows]); y=np.asarray([float(r['f1_loss_pp']) for r in rows])
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x)==0:
            raise ValueError('invalid or constant overlap values')
        slope,intercept=np.polyfit(x,y,1);r=float(np.corrcoef(x,y)[0,1]) if np.ptp(y)>0 else None
        style.setup();fig,ax=plt.subplots(figsize=(style.TEXT_WIDTH,3.2),constrained_layout=True)
        markers={'PER':'o','LOC':'s','ORG':'^'}
        targets=list(dict.fromkeys(row['target'] for row in rows))
        colors={target:plt.get_cmap('tab10')(i%10) for i,target in enumerate(targets)}
        for row in rows:
            ax.scatter(100*row['overlap'],row['f1_loss_pp'],marker=markers[row['entity_type']],color=colors[row['target']],s=36)
        grid=np.asarray([x.min(),x.max()]);ax.plot(grid,intercept+slope*grid,'--',color='#333333')
        ax.set_xlabel('Target mentions seen in source training (%)');ax.set_ylabel('M2 F1 loss vs. source test (pp)')
        if r is not None:ax.text(.98,.03,f'r={r:.2f}; n={len(rows)}',transform=ax.transAxes,ha='right')
        handles=[Line2D([0],[0],marker='o',linestyle='',color=colors[t],label=t) for t in targets]
        handles += [Line2D([0],[0],marker=m,linestyle='',color='#777777',label=k) for k,m in markers.items()]
        ax.legend(handles=handles,ncol=2,fontsize=7)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.metadata.parent.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=args.output.parent) as td:
            staged=Path(td)/'figure.pdf';fig.savefig(staged,format='pdf')
            with args.output.open('xb') as out,staged.open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):out.write(block)
        plt.close(fig)
        metadata={'input':str(args.input),'input_sha256':sha(args.input),'frozen_manifest_sha256':frozen,
                  'source':args.source,'method':'M2 (A analysis)','label_space':'LOC+ORG+PER',
                  'points':len(rows),'pearson_r':r,'ols_slope_loss_pp_per_overlap_pp':float(slope),
                  'ols_intercept_pp':float(intercept),'overlap_definition':required,
                  'loss_definition':report['loss_definition'],
                  'interpretation':'Descriptive association; points share datasets and entity types and do not establish causality.',
                  'figure_sha256':sha(args.output)}
        with args.metadata.open('x',encoding='utf-8') as out:json.dump(metadata,out,indent=2);out.write('\n')
        print(f'F6(b): M2 source={args.source}, points={len(rows)}, Pearson r={r}')
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(2,f'Error: {error}\n')


if __name__=='__main__':main()
