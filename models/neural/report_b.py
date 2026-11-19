"""Generate B's T3 LaTeX tables and English section draft from compatible CSV rows.

Only precomputed CSV records are read. Missing metrics become explicit TODOs;
validation results are never used as test results. Existing files are not replaced.
"""
import argparse
import statistics
from pathlib import Path

from eval.result_tables import read_csv
from models.llm.common import TYPE_DESCRIPTIONS
from models.neural.run_b_evaluation import load_frozen

NATIVE = ('conll2003', 'wnut17', 'mit_restaurant')
TARGETS = ('conll2003', 'conllpp', 'wikigold', 'btc', 'wnut17', 'mit_restaurant')
DISPLAY = {'conll2003': 'CoNLL-03', 'conllpp': 'CoNLL++', 'wikigold': 'WikiGold',
           'btc': 'BTC', 'wnut17': 'WNUT-17', 'mit_restaurant': 'MIT Rest.'}


def label_space(dataset):
    if dataset == 'btc':
        return 'LOC+ORG+PER'
    return '+'.join(sorted(TYPE_DESCRIPTIONS[dataset]))


def score(rows, method, dataset, scope, metric, seed):
    source = dataset if dataset in NATIVE else 'conll2003'
    matching = [r for r in rows if r['experiment'] == 'baseline' and r['condition'] == 'standard'
                and r['method'] == method and r['dataset'] == dataset and (r['member'] == 'C' or r['train_source'] == source)
                and r['split'] == 'test' and r['scope'] == scope and r['label_space'] == label_space(dataset)
                and r['entity_type'] == 'all' and r['entity_group'] in ('all', '') and r['seed'] == str(seed)
                and r['metric'] == metric]
    if len(matching) > 1:
        raise ValueError(f'duplicate {method}/{dataset}/{scope}/{metric}/seed{seed}')
    if not matching:
        return None
    value = float(matching[0]['value'])
    if not 0 <= value <= 1:
        raise ValueError(f'non-probability result: {method}/{dataset}/{metric}')
    return value


def fmt(value):
    return r'\textbf{TODO}' if value is None else f'{value * 100:.1f}'


def make_tables(rows, methods, seed, c_methods=()):
    full = [r'% Requires booktabs and graphicx. Seed 0; exact-entity metrics in percent.',
            r'\begin{tabular}{@{}llrrrrr@{}}', r'\toprule',
            r'Dataset & Method & P & R & F1 & CI low & CI high \\', r'\midrule']
    missing = 0
    for dataset in NATIVE:
        for method in methods:
            vals = [score(rows, method, dataset, 'full', metric, seed)
                    for metric in ('precision', 'recall', 'f1', 'f1_ci95_lower', 'f1_ci95_upper')]
            if any(v is None for v in vals):
                missing += 1
            full.append(f"{DISPLAY[dataset]} & {method} & " + ' & '.join(fmt(v) for v in vals) + r' \\')
    full.extend([r'\bottomrule', r'\end{tabular}', ''])
    subset = [r'% Fixed 300 sentences per target. BTC uses PER/LOC/ORG; other columns use their native labels.',
              r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
              'Method & ' + ' & '.join(DISPLAY[x] for x in TARGETS) + r' \\', r'\midrule']
    for method in (*methods, *c_methods):
        vals = [score(rows, method, dataset, 'test_300', 'f1', seed) for dataset in TARGETS]
        missing += sum(value is None for value in vals)
        subset.append(method + ' & ' + ' & '.join(fmt(value) for value in vals) + r' \\')
    subset.extend([r'\bottomrule', r'\end{tabular}', ''])
    return '\n'.join(full), '\n'.join(subset), missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, nargs='+', required=True)
    parser.add_argument('--c-csv', type=Path, help='C fixed-300 records independently verified against frozen B inputs')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=('M3', 'M4', 'Span'))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--frozen', type=Path, help='Frozen dev-only primary-model selection')
    parser.add_argument('--latex-prefix', default='b_handoff', help='Relative path after copying into main LaTeX project')
    args = parser.parse_args()
    try:
        if len(set(args.methods)) != len(args.methods) or not set(args.methods) <= {'M0','M1','M2','M3','M4','Span'}:
            raise ValueError('methods must be distinct supervised identifiers M0–M4 or Span')
        if args.output_dir.exists():
            raise FileExistsError(f'output directory exists: {args.output_dir}')
        rows = [row for path in args.csv for row in read_csv(path)]
        c_methods = ('GLiNER', 'Qwen1.5B-zero', 'Qwen7B-zero', 'Qwen7B-few') if args.c_csv else ()
        if args.c_csv:
            c_rows=list(read_csv(args.c_csv))
            if any(row['member'] != 'C' or row['method'] not in c_methods or row['split'] != 'test' or row['scope'] != 'test_300' for row in c_rows):
                raise ValueError('C input contains an unsupported method or scope')
            rows.extend(c_rows)
        full, subset, missing = make_tables(rows, args.methods, args.seed, c_methods)
        seed_table = [r'% Neural seed variability: mean and sample SD on the same native full test.',
                      r'\begin{tabular}{@{}llrr@{}}', r'\toprule',
                      r'Dataset & Method & Mean F1 & Sample SD \\', r'\midrule']
        for dataset in NATIVE:
            for method in ('M3','M4','Span'):
                vals=[score(rows,method,dataset,'full','f1',seed) for seed in range(3)]
                if any(value is None for value in vals):
                    mean=sd=r'\textbf{TODO}'
                else:
                    mean=f'{statistics.mean(vals)*100:.1f}'; sd=f'{statistics.stdev(vals)*100:.1f}'
                seed_table.append(f'{DISPLAY[dataset]} & {method} & {mean} & {sd} ' + r'\\')
        seed_table.extend([r'\bottomrule',r'\end{tabular}',''])
        if args.frozen:
            selected=load_frozen(args.frozen)['selection']['primary_method']
            p,r,f,lo,hi=[score(rows,selected,'conll2003','full',metric,0) for metric in
                         ('precision','recall','f1','f1_ci95_lower','f1_ci95_upper')]
            if any(x is None for x in (p,r,f,lo,hi)):
                primary='The validation-selected primary extractor is '+selected+r'; its seed-0 held-out P/R/F1 and interval are \textbf{TODO}.'
            else:
                primary=(f'The validation-selected primary extractor is {selected}; on CoNLL-2003 test, seed 0 obtains '
                         f'P={p*100:.2f}\\%, R={r*100:.2f}\\%, F1={f*100:.2f}\\%, '
                         f'with a sentence-bootstrap 95\\% F1 interval [{lo*100:.2f}, {hi*100:.2f}]\\%.')
        else:
            primary=r'TODO: Insert the validation-selected primary method and its held-out P/R/F1 and confidence interval.'
        prefix = args.latex_prefix.rstrip('/')
        draft = r'''% Requires graphicx and booktabs. Review and adapt section numbering before integration.
\section{Supervised extraction methods}
We study a character-CNN BiLSTM--CRF (M3), a fine-tuned BERT token classifier (M4), and a span classifier. The BiLSTM--CRF combines pretrained word representations and character features with structured sequence decoding. The BERT token classifier predicts BIO labels for the original words using contextual representations. The span classifier scores candidate boundaries and types. Exact architectural and hyperparameter details are documented in the accompanying frozen configurations. TODO: describe each model's training objective and final hyperparameters.

\section{Main supervised results}
The primary method is chosen using full CoNLL-2003 validation micro-F1, before applying this automation to held-out data. Three seeds are available for neural models; the following tables display representative seed 0, while seed means and sample standard deviations must be reported separately. The repository contained A/C held-out outputs before this automation; this limits any claim that the entire project team was blinded to test results.
PRIMARY

\begin{table}[t]\centering
\caption{Native-domain full test results. Percentages are exact-entity precision, recall, F1 and the sentence-bootstrap 95\% F1 interval. TODO marks unavailable results.}
\input{PREFIX/t3_b_native_full.tex}
\end{table}
\begin{table}[t]\centering
\caption{Fixed 300-sentence test subsets. BTC uses common PER/LOC/ORG; other datasets use native categories. Results across label schemas are not directly interchangeable.}
\input{PREFIX/t3_b_300.tex}
\end{table}
\begin{table}[t]\centering
\caption{Neural full-test seed means and sample standard deviations. TODO denotes an incomplete group.}
\input{PREFIX/t3_b_seed_summary.tex}
\end{table}
\IfFileExists{PREFIX/figures/f2_b.pdf}{\includegraphics[width=\linewidth]{PREFIX/figures/f2_b.pdf}}{\textbf{TODO: add F2.}}
Adjacent-method changes in F2 are descriptive differences, not single-factor causal effects. TODO: report paired sentence-bootstrap intervals and all-seed variability.

\section{Where does supervised performance come from?}
\IfFileExists{PREFIX/figures/f3b_b.pdf}{\includegraphics[width=\linewidth]{PREFIX/figures/f3b_b.pdf}}{\textbf{TODO: add F3(b).}}
TODO: compare the four input representations and discuss variability and controls. F4 uses a deterministic sample of at most 1,000 matched validation entities for its t-SNE panels; the separate MISC cluster worksheet uses the complete matched validation intersection, so its sample is not identical to the plotted subset. TODO: review and interpret MISC subclusters manually, then add K-means/ARI/NMI findings with limitations. TODO: add boundary-error analysis comparing M4 and Span.

\section{Cross-domain generalization}
\IfFileExists{PREFIX/figures/f6a_b.pdf}{\includegraphics[width=\linewidth]{PREFIX/figures/f6a_b.pdf}}{\textbf{TODO: add F6(a).}}
\IfFileExists{PREFIX/figures/f6b_from_a_m2.pdf}{\includegraphics[width=\linewidth]{PREFIX/figures/f6b_from_a_m2.pdf}}{\textbf{TODO: add A's M2 overlap analysis as F6(b).}}
The transfer matrix uses the shared PER/LOC/ORG label space; the native-label results above are a different evaluation. F6(b) uses A's M2 results rather than BERT: the horizontal axis is the fraction of target gold mentions whose case-folded full name and mapped type occur in the source training set, while the vertical axis is per-type F1 loss from the source-domain test. The plotted regression is descriptive and its source/target/type points share data. TODO: report the correlation and sample size from the checked F6(b) metadata and quantify uncertainty. TODO: incorporate perturbation and replacement results only after their checked paired evaluations are complete.
'''.replace('PREFIX', prefix).replace('PRIMARY', primary)
        args.output_dir.mkdir(parents=True)
        for name, content in (('t3_b_native_full.tex', full), ('t3_b_300.tex', subset),
                              ('t3_b_seed_summary.tex','\n'.join(seed_table)), ('b_sections.tex', draft)):
            with (args.output_dir / name).open('x', encoding='utf-8') as target:
                target.write(content)
        print(f'Wrote LaTeX files to {args.output_dir}; missing table cells/groups: {missing}')
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
