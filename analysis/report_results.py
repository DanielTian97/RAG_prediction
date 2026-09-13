"""Export Tables 1/2 and Figure 3 data from the final ECIR summaries.

Derived from final_analysis_v3.ipynb. Does not refit models. Significance flags
preserve the notebook helper and its additional PerpA comparison for PostGen;
these flags are labelled as notebook results, not independently validated tests.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETRIEVERS = ['bm25', 'mt5', 'e5']
METHODS = {'nqc': 'NQC', 'maxScore': 'MaxScore', 'spatial': 'DenseQPP',
           'a_ratio': 'A-Pair-Ratio', 'bertQPP': 'BERT-QPP'}
COMBINATIONS = ['p0', 'p01', 'p013', 'p0123']


def one(frame, column):
    if len(frame) != 1:
        raise ValueError(f'Expected one result for {column}, found {len(frame)}; check missing/duplicate rows.')
    return frame.iloc[0][column]


def build_tables(single, union):
    import pandas as pd
    from tools.fisher_test import compare_spearman_rhos

    def score(method, target, ret, k=2):
        return float(one(single[(single.task == 'nq') & (single.k == k) &
                                (single.target_metric == target) & (single.retriever == ret) &
                                (single.QPP_Method == method)], 'Spearman'))

    def ensemble(task, ret, combination, post, k=2):
        return union[(union['QA Task'] == 'nq') & (union['Top-Retrieved Docs'] == k) &
                     (union['Prediction Name'] == task) & (union.Retriever == ret) &
                     (union.Combination_Number == combination) & (union.Use_Postgen == post)]

    table1 = []
    for method, label in METHODS.items():
        row = {'Method': label}
        for task, target in [('RPP','utility'), ('GPP','f1')]:
            for ret in RETRIEVERS:
                row[f'{task}_{ret}'] = score(method, target, ret)
        table1.append(row)

    table2, flags = [], []
    for post, combinations in [(False, COMBINATIONS), (True, ['perpa'] + COMBINATIONS)]:
        for combination in combinations:
            row = {'Stage': 'PostGen' if post else 'PreGen', 'Combination': combination}
            for task, target in [('RPP','utility'), ('GPP','f1')]:
                for ret in RETRIEVERS:
                    label = f'{task}_{ret}'
                    if combination == 'perpa':
                        row[label] = score('prob(k)', target, ret)
                        continue
                    frame = ensemble(task, ret, combination, post)
                    value = float(one(frame, 'Rho'))
                    n = int(one(frame, 'Number of Queries'))
                    baseline = float(one(ensemble(task, ret, combination if post else 'p0', False), 'Rho'))
                    def significant(reference):
                        result = compare_spearman_rhos(n, reference, value)
                        return bool(result['z_stat'] < 0 and result['p_value'] < .05)
                    flag = significant(baseline)
                    if post:
                        flag = flag and significant(score('prob(k)', target, ret))
                    row[label] = value
                    flags.append({'Stage':row['Stage'], 'Combination':combination,
                                  'Task':task, 'Retriever':ret, 'Queries':n,
                                  'Notebook_significant':flag})
            table2.append(row)

    figure = []
    for task, target in [('RPP','utility'), ('GPP','f1')]:
        for k in [2,3,5,7,10]:
            values = {'A-Pair-Ratio':score('a_ratio', target, 'e5', k),
                      'PerpA':score('prob(k)', target, 'e5', k),
                      'PreGen-Ensmbl':float(one(ensemble(task,'e5','p0123',False,k),'Rho')),
                      'PostGen-Ensmbl':float(one(ensemble(task,'e5','p0123',True,k),'Rho'))}
            for method, value in values.items():
                figure.append({'Task':task, 'k':k, 'Method':method, 'Rho':value})
    return tuple(pd.DataFrame(rows) for rows in [table1, table2, flags, figure])


def plot_figure(data, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=True)
    order = ['A-Pair-Ratio','PreGen-Ensmbl','PerpA','PostGen-Ensmbl']
    for ax, task, prefix in zip(axes, ['RPP','GPP'], ['(a)','(b)']):
        for method in order:
            rows = data[(data.Task == task) & (data.Method == method)].sort_values('k')
            ax.plot(rows.k, rows.Rho, marker='o', label=method)
        ax.set(title=f'{prefix} {task} Accuracy', xlabel='Context Size (k)',
               xticks=[2,3,5,7,10], ylim=(.11,.45), yticks=[.2,.3,.4])
        ax.grid(True, linestyle='--', alpha=.6)
    axes[0].set_ylabel("Spearman's ρ")
    axes[1].legend(loc='upper left', bbox_to_anchor=(1,1))
    fig.tight_layout()
    for suffix in ['pdf','svg','png']:
        fig.savefig(output/f'figure3.{suffix}', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'analysis/ecir_res')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new or empty output directory.')
    import pandas as pd
    single = pd.read_csv(args.results_dir/'single_output_v2.csv')
    union = pd.read_csv(args.results_dir/'union_output_v2.csv')
    tables = build_tables(single, union)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in zip(['table1','table2','notebook_significance','figure3_data'], tables):
        table.to_csv(args.output_dir/f'{name}.csv', index=False, float_format='%.10g')
        if name in ['table1','table2']:
            lines = [r'\begin{tabular}{' + 'l' * len(table.columns) + '}', r'\hline']
            escape = lambda value: str(value).replace('_', r'\_')
            lines.append(' & '.join(map(escape, table.columns)) + r' \\')
            for row in table.itertuples(index=False, name=None):
                lines.append(' & '.join(f'{v:.4f}' if isinstance(v, float) else escape(v) for v in row) + r' \\')
            lines.extend([r'\hline', r'\end{tabular}'])
            (args.output_dir/f'{name}.tex').write_text('\n'.join(lines) + '\n')
    if not args.no_plot:
        plot_figure(tables[-1], args.output_dir)
    print(f'Exported Tables 1/2 and 40 Figure 3 data points to {args.output_dir}')


if __name__ == '__main__':
    main()
