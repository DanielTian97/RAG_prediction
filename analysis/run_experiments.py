"""ECIR experiment pipeline extracted from easy_analysis_readability-v2.ipynb.

Preserves feature transformations, filtering, regression and correlations.
Requires external inputs; use --check-inputs before running. NQ is the default;
--tasks nq dl also retains the source notebook's historical DL experiments.
"""
import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--material-dir', type=Path, default=ROOT.parent / 'rag_utility')
    parser.add_argument('--output-dir', type=Path, required=True, help='New or empty output directory')
    parser.add_argument('--context-sizes', nargs='+', type=int, choices=[2,3,5,7,10], default=[2,3,5,7,10])
    parser.add_argument('--retrievers', nargs='+', choices=['bm25','mt5','e5'], default=['bm25','mt5','e5'])
    parser.add_argument('--tasks', nargs='+', choices=['nq','dl'], default=['nq'])
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    args.material_dir = args.material_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    for name in ('context_sizes','retrievers','tasks'):
        setattr(args, name, list(dict.fromkeys(getattr(args, name))))
    return args


def required_inputs(args):
    paths = set()
    for task in args.tasks:
        splits = ['nq_dev','nq_test'] if task == 'nq' else ['dev_small','dl']
        prefix, suffix = ('short','concise') if task == 'nq' else ('random','prompt1')
        for split in splits:
            calls = 5 if split == 'dl' else 1
            eval_splits = ['19','20'] if split == 'dl' else [split]
            for ret in args.retrievers:
                for k in args.context_sizes:
                    for part in eval_splits:
                        zero = f'{prefix}_answers_0shot_{calls}calls_0_0_bm25_dl_{part}_{suffix}_eval.json'
                        stem = f'{prefix}_answers_{k}shot_{calls}calls_1_0_{ret}_dl_{part}_{suffix}'
                        paths.update([args.material_dir/'eval_results'/zero,
                                      args.material_dir/'eval_results'/f'{stem}_eval.json',
                                      args.material_dir/'gen_results'/f'{stem}.json'])
                    paths.update([
                        ROOT/f'analysis/precomputed_qpps/{ret}_{k}_combined_qpp_{split}.csv',
                        ROOT/f'perplexity_eval/log_prob_temp_res/full_context_with_query/{split}_{ret}_{k}.json',
                        ROOT/f'perplexity_eval/log_prob_temp_res/individual_with_query/{split}_{ret}_20.json',
                        ROOT/f'qualt5_eval/quality_res/{ret}_{split}.csv',
                        ROOT/f'qualt5_eval/quality_res/{ret}_{split}_integrated_{k}.csv',
                        ROOT/f'readability_eval/readability_res/individual_readability_{split}_{ret}_top_10.csv',
                        ROOT/f'readability_eval/readability_res/integrated_readability_{split}_{ret}_top_{k}.csv'])
    return sorted(paths)


def run(args):
    import pandas as pd
    import json
    import numpy as np
    from tools import process_results
    from scipy import stats
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn import linear_model
    import itertools
    import pathlib
    import copy
    from tqdm import tqdm

    def concate_str(l):
        o = ''
        for li in l:
            o += f'_{li}'
        return o[1:]



    output_path = str(args.output_dir / "union_output_v2.csv")
    single_output_path = str(args.output_dir / "single_output_v2.csv")

    for _k, _ret, _task, target_metric in tqdm(itertools.product(args.context_sizes, args.retrievers, args.tasks, ['f1', 'utility'])):
    
        print(_k, _ret, _task, target_metric)

        
        if(_task=='nq'):
            _dataset_dev, _dataset_test, _prefix, _suffix = 'nq_dev', ['nq_test'], 'short', 'concise'
        elif(_task=='dl'):
            _dataset_dev, _dataset_test, _prefix, _suffix = 'dev_small', ['19', '20'], 'random', 'prompt1'
    
        # loading dev data
        print('~loading dev data')
    
        zero_evals = process_results.load_json(f'{args.material_dir}/eval_results/{_prefix}_answers_0shot_1calls_0_0_bm25_dl_{_dataset_dev}_{_suffix}_eval.json')
        k_evals = process_results.load_json(f'{args.material_dir}/eval_results/{_prefix}_answers_{_k}shot_1calls_1_0_{_ret}_dl_{_dataset_dev}_{_suffix}_eval.json')
        k_gens = process_results.load_json(f'{args.material_dir}/gen_results/{_prefix}_answers_{_k}shot_1calls_1_0_{_ret}_dl_{_dataset_dev}_{_suffix}.json')
        perpC = process_results.load_json(f'{ROOT / "perplexity_eval"}/log_prob_temp_res/full_context_with_query/{_dataset_dev}_{_ret}_{_k}.json')
        perpC_indi = process_results.load_json(f'{ROOT / "perplexity_eval"}/log_prob_temp_res/individual_with_query/{_dataset_dev}_{_ret}_20.json')
    
        top_k_perpC_dict = {i[0]: [sub_i[1] for sub_i in i[1].items() if int(sub_i[0])<_k] for i in perpC_indi.items()}
        avg_indi_perpC_dict = {i[0]:np.mean(i[1]) for i in top_k_perpC_dict.items()}
        max_indi_perpC_dict = {i[0]:np.max(i[1]) for i in top_k_perpC_dict.items()}
        min_indi_perpC_dict = {i[0]:np.min(i[1]) for i in top_k_perpC_dict.items()}
    
        readability_res_path = f'{ROOT / "readability_eval"}/readability_res/individual_readability_{_dataset_dev}_{_ret}_top_10.csv'
        readability_res_path_itg = f'{ROOT / "readability_eval"}/readability_res/integrated_readability_{_dataset_dev}_{_ret}_top_{_k}.csv'
        available_metrics = pd.read_csv(readability_res_path).readability_metric.unique()
        readability_max_dict, readability_min_dict, readability_avg_dict, readability_itg_dict = {}, {}, {}, {}
        for _r_m in available_metrics:
            _r_m_max, _r_m_avg, _r_m_min = process_results.process_qualt5_res_individual(readability_res_path, _k, _r_m)
            _r_m_itg = process_results.process_qualt5_res_integrated(readability_res_path_itg, _r_m)
            readability_max_dict.update({_r_m: copy.deepcopy(_r_m_max)})
            readability_min_dict.update({_r_m: copy.deepcopy(_r_m_min)})
            readability_avg_dict.update({_r_m: copy.deepcopy(_r_m_avg)})
            readability_itg_dict.update({_r_m: copy.deepcopy(_r_m_itg)})
    
        doc_qual_max, doc_qual_avg, doc_qual_min = process_results.process_qualt5_res_individual(f'{ROOT / "qualt5_eval"}/quality_res/{_ret}_{_dataset_dev}.csv', _k)
        doc_qual_itg = process_results.process_qualt5_res_integrated(f'{ROOT / "qualt5_eval"}/quality_res/{_ret}_{_dataset_dev}_integrated_{_k}.csv')
    
        qpp_df = pd.read_csv(f'{ROOT / "analysis/precomputed_qpps"}/{_ret}_{_k}_combined_qpp_{_dataset_dev}.csv')
    
        dev_res = qpp_df[['qid', 'query']].drop_duplicates().copy()
    
        # Expand the dataframe for the convenience of analysis
        for qpp_name in qpp_df.qpp_method.unique():
            value_dict = dict(zip(qpp_df[qpp_df.qpp_method==qpp_name]['qid'], qpp_df[qpp_df.qpp_method==qpp_name]['qpp_estimate']))
            dev_res[qpp_name] = dev_res.qid.apply(lambda _qid: value_dict[_qid])
    
        if(_task=='nq'):
            base_f1_dict = {item[0]: item[1]['0']['0']['F1'] for item in zero_evals.items()}
            f1_dict = {item[0]: item[1]['0']['0']['F1'] for item in k_evals.items()}
            kshot_prob_dict = {item[0]: np.mean(ast.literal_eval(item[1]['0']['0']['probs'])) for item in k_gens.items()}
        elif(_task=='dl'):
            base_f1_dict = {item[0]: item[1]['0']['0']['qrel_1']['f1']['max'] for item in zero_evals.items() if ('0' in item[1].keys())}
            f1_dict = {item[0]: item[1]['0']['0']['qrel_1']['f1']['max'] for item in k_evals.items() if ('0' in item[1].keys())}
            kshot_prob_dict = {item[0]: np.mean(ast.literal_eval(item[1]['0']['0']['probs'])) for item in k_gens.items() if ('0' in item[1].keys())}
    
        dev_res = dev_res[dev_res.qid.astype('str').isin(f1_dict.keys())]
        dev_res['f1'] = dev_res.qid.apply(lambda x: f1_dict[str(x)])
        dev_res['utility'] = dev_res.qid.apply(lambda x: f1_dict[str(x)]-base_f1_dict[str(x)])
        dev_res['prob(k)'] = dev_res.qid.apply(lambda x: kshot_prob_dict[str(x)])
    
        dev_res['max(perpC)'] = dev_res.qid.apply(lambda x: max_indi_perpC_dict[str(x)])
        dev_res['min(perpC)'] = dev_res.qid.apply(lambda x: min_indi_perpC_dict[str(x)])
        dev_res['avg(perpC)'] = dev_res.qid.apply(lambda x: avg_indi_perpC_dict[str(x)])
        dev_res['itg(perpC)'] = dev_res.qid.apply(lambda x: perpC[str(x)])
    
        dev_res['max(docQual)'] = dev_res.qid.apply(lambda x: doc_qual_max[str(x)])
        dev_res['min(docQual)'] = dev_res.qid.apply(lambda x: doc_qual_min[str(x)])
        dev_res['avg(docQual)'] = dev_res.qid.apply(lambda x: doc_qual_avg[str(x)])
        dev_res['itg(docQual)'] = dev_res.qid.apply(lambda x: doc_qual_itg[str(x)])
    
        dev_res = dev_res[dev_res.qid.astype('str').isin(readability_itg_dict['Spache'].keys())]
    
        readability_cols = []
        for _r_m in available_metrics:
            readability_cols += [f'max({_r_m})', f'min({_r_m})', f'avg({_r_m})', f'itg({_r_m})']
    
            for qid in dev_res.qid.unique():
                if qid not in readability_max_dict[_r_m].keys():
                    readability_max_dict[_r_m].update({str(qid): -1})
                    readability_min_dict[_r_m].update({str(qid): -1})
                    readability_avg_dict[_r_m].update({str(qid): -1})
                if qid not in readability_itg_dict[_r_m].keys():
                    readability_itg_dict[_r_m].update({str(qid): -1})
        
            dev_res[f'max({_r_m})'] = dev_res.qid.apply(lambda x: readability_max_dict[_r_m][str(x)])
            dev_res[f'min({_r_m})'] = dev_res.qid.apply(lambda x: readability_min_dict[_r_m][str(x)])
            dev_res[f'avg({_r_m})'] = dev_res.qid.apply(lambda x: readability_avg_dict[_r_m][str(x)])
            dev_res[f'itg({_r_m})'] = dev_res.qid.apply(lambda x: readability_itg_dict[_r_m][str(x)])

        dropped_columns_dev = [col for col in readability_cols if (dev_res[col] == -1).any()]
        dev_res = dev_res.drop(columns=[col for col in readability_cols if (dev_res[col] == -1).any()])
        dev_res = dev_res.dropna(axis=1)
        readability_cols = [e for e in readability_cols if e not in dropped_columns_dev]
    
        dev_res = dev_res.dropna(axis='index')
    
        # loading test data
        print('~loading test data')
        zero_evals, k_evals, k_gens, perpC  = {}, {}, {}, {}
    
        _calls = 5 if _task=='dl' else 1
    
        for _d in _dataset_test:
            zero_evals.update(process_results.load_json(f'{args.material_dir}/eval_results/{_prefix}_answers_0shot_{_calls}calls_0_0_bm25_dl_{_d}_{_suffix}_eval.json'))
            k_evals.update(process_results.load_json(f'{args.material_dir}/eval_results/{_prefix}_answers_{_k}shot_{_calls}calls_1_0_{_ret}_dl_{_d}_{_suffix}_eval.json'))
            k_gens.update(process_results.load_json(f'{args.material_dir}/gen_results/{_prefix}_answers_{_k}shot_{_calls}calls_1_0_{_ret}_dl_{_d}_{_suffix}.json'))
    
        if(_task == 'dl'):
            _d = 'dl'
        else:
            _d = 'nq_test'
    
        perpC = process_results.load_json(f'{ROOT / "perplexity_eval"}/log_prob_temp_res/full_context_with_query/{_d}_{_ret}_{_k}.json')
        perpC_indi = process_results.load_json(f'{ROOT / "perplexity_eval"}/log_prob_temp_res/individual_with_query/{_d}_{_ret}_20.json')
    
        top_k_perpC_dict = {i[0]: [sub_i[1] for sub_i in i[1].items() if int(sub_i[0])<_k] for i in perpC_indi.items()}
        avg_indi_perpC_dict = {i[0]:np.mean(i[1]) for i in top_k_perpC_dict.items()}
        max_indi_perpC_dict = {i[0]:np.max(i[1]) for i in top_k_perpC_dict.items()}
        min_indi_perpC_dict = {i[0]:np.min(i[1]) for i in top_k_perpC_dict.items()}
    
        readability_res_path = f'{ROOT / "readability_eval"}/readability_res/individual_readability_{_d}_{_ret}_top_10.csv'
        readability_res_path_itg = f'{ROOT / "readability_eval"}/readability_res/integrated_readability_{_d}_{_ret}_top_{_k}.csv'
        available_metrics = pd.read_csv(readability_res_path).readability_metric.unique()
        readability_max_dict, readability_avg_dict, readability_itg_dict, readability_min_dict = {}, {}, {}, {}
        for _r_m in available_metrics:
            _r_m_max, _r_m_avg, _r_m_min = process_results.process_qualt5_res_individual(readability_res_path, _k, _r_m)
            _r_m_itg = process_results.process_qualt5_res_integrated(readability_res_path_itg, _r_m)
            readability_max_dict.update({_r_m: copy.deepcopy(_r_m_max)})
            readability_min_dict.update({_r_m: copy.deepcopy(_r_m_min)})
            readability_avg_dict.update({_r_m: copy.deepcopy(_r_m_avg)})
            readability_itg_dict.update({_r_m: copy.deepcopy(_r_m_itg)})
    
        doc_qual_max, doc_qual_avg, doc_qual_min = process_results.process_qualt5_res_individual(f'{ROOT / "qualt5_eval"}/quality_res/{_ret}_{_d}.csv', _k)
        doc_qual_itg = process_results.process_qualt5_res_integrated(f'{ROOT / "qualt5_eval"}/quality_res/{_ret}_{_d}_integrated_{_k}.csv')
    
        qpp_df = pd.read_csv(f'{ROOT / "analysis/precomputed_qpps"}/{_ret}_{_k}_combined_qpp_{_d}.csv')
    
        test_res = qpp_df[['qid', 'query']].drop_duplicates().copy()
    
        # Expand the dataframe for the convenience of analysis
        for qpp_name in qpp_df.qpp_method.unique():
            value_dict = dict(zip(qpp_df[qpp_df.qpp_method==qpp_name]['qid'], qpp_df[qpp_df.qpp_method==qpp_name]['qpp_estimate']))
            test_res[qpp_name] = test_res.qid.apply(lambda _qid: value_dict[_qid])
    
        if(_task=='nq'):
            base_f1_dict = {item[0]: item[1]['0']['0']['F1'] for item in zero_evals.items()}
            f1_dict = {item[0]: item[1]['0']['0']['F1'] for item in k_evals.items()}
            kshot_prob_dict = {item[0]: np.mean(ast.literal_eval(item[1]['0']['0']['probs'])) for item in k_gens.items()}
        elif(_task=='dl'):
            base_f1_dict = {item[0]: process_results.tool_for_aggregating_dl_performance(item) for item in zero_evals.items() if ('0' in item[1].keys())}
            f1_dict = {item[0]: process_results.tool_for_aggregating_dl_performance(item) for item in k_evals.items() if ('0' in item[1].keys())}
            kshot_prob_dict = {item[0]: np.mean(ast.literal_eval(item[1]['0']['0']['probs'])) for item in k_gens.items() if ('0' in item[1].keys())}
    
        test_res = test_res[test_res.qid.astype('str').isin(f1_dict.keys())]
        test_res['f1'] = test_res.qid.apply(lambda x: f1_dict[str(x)])
        test_res['utility'] = test_res.qid.apply(lambda x: f1_dict[str(x)]-base_f1_dict[str(x)])
        test_res['prob(k)'] = test_res.qid.apply(lambda x: kshot_prob_dict[str(x)])
    
        test_res['max(perpC)'] = test_res.qid.apply(lambda x: max_indi_perpC_dict[str(x)])
        test_res['min(perpC)'] = test_res.qid.apply(lambda x: min_indi_perpC_dict[str(x)])
        test_res['avg(perpC)'] = test_res.qid.apply(lambda x: avg_indi_perpC_dict[str(x)])
        test_res['itg(perpC)'] = test_res.qid.apply(lambda x: perpC[str(x)])
    
        test_res['max(docQual)'] = test_res.qid.apply(lambda x: doc_qual_max[str(x)])
        test_res['min(docQual)'] = test_res.qid.apply(lambda x: doc_qual_min[str(x)])
        test_res['avg(docQual)'] = test_res.qid.apply(lambda x: doc_qual_avg[str(x)])
        test_res['itg(docQual)'] = test_res.qid.apply(lambda x: doc_qual_itg[str(x)])
    
        test_res = test_res[test_res.qid.astype('str').isin(readability_itg_dict['Spache'].keys())]
    
        readability_cols = []
        for _r_m in available_metrics:
            readability_cols += [f'max({_r_m})', f'min({_r_m})', f'avg({_r_m})', f'itg({_r_m})']
    
            for qid in test_res.qid.unique():
                if qid not in readability_max_dict[_r_m].keys():
                    readability_max_dict[_r_m].update({str(qid): -1})
                    readability_min_dict[_r_m].update({str(qid): -1})
                    readability_avg_dict[_r_m].update({str(qid): -1})
                if qid not in readability_itg_dict[_r_m].keys():
                    readability_itg_dict[_r_m].update({str(qid): -1})
        
            test_res[f'max({_r_m})'] = test_res.qid.apply(lambda x: readability_max_dict[_r_m][str(x)])
            test_res[f'min({_r_m})'] = test_res.qid.apply(lambda x: readability_min_dict[_r_m][str(x)])
            test_res[f'avg({_r_m})'] = test_res.qid.apply(lambda x: readability_avg_dict[_r_m][str(x)])
            test_res[f'itg({_r_m})'] = test_res.qid.apply(lambda x: readability_itg_dict[_r_m][str(x)])
    
        dropped_columns = [col for col in readability_cols if (test_res[col] == -1).any()]
        test_res = test_res.drop(columns=[col for col in readability_cols if (test_res[col] == -1).any()])
        test_res = test_res.dropna(axis=1)
        readability_cols = [e for e in readability_cols if e not in dropped_columns]
        readability_cols = [e for e in readability_cols if e not in dropped_columns_dev]
    
        print('~keep dev and test data having the same features')
        common_cols = test_res.columns.intersection(dev_res.columns)
    
        test_res = test_res[common_cols]
        dev_res = dev_res[common_cols]
    
        print('~correlation between predictors')
        corr = test_res.drop(columns=['qid', 'query', 'f1', 'utility', 'bertQPP(QV)']).corr(method="spearman")
    
        # Plot heatmap using matplotlib
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(corr, cmap="coolwarm")
    
        # Add colorbar
        fig.colorbar(cax)
    
        # Set ticks and labels
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="left")
        ax.set_yticklabels(corr.columns)
    
        plt.title(f'{_task}_{_ret}_{_k}', pad=20)
        plt.savefig(f"{args.output_dir / "heatmaps"}/heatmap_{_task}_{_ret}_{_k}_v2.png", format="png", bbox_inches="tight")  
        plt.savefig(f"{args.output_dir / "heatmaps"}/heatmap_{_task}_{_ret}_{_k}_v2.pdf", format="pdf", bbox_inches="tight")  
        plt.close(fig)
    
        # single correlations
        df_content = []
        for predictor_name in corr:
            df_content.append([predictor_name, stats.spearmanr(test_res[predictor_name], test_res[target_metric])[0], stats.kendalltau(test_res[predictor_name], test_res[target_metric])[0]])
    

        a1 = pd.DataFrame(df_content, columns=['QPP_Method', 'Spearman', 'Kendall'])
        a1.Spearman = a1.Spearman.apply(lambda x: round(x, 4))
        a1.Kendall = a1.Kendall.apply(lambda x: round(x, 4))

        single_csvfile = pathlib.Path(single_output_path)
        single_df_to_write = pd.concat([pd.DataFrame(np.tile(np.array([_k, _ret, _task, target_metric]), (len(a1), 1)), columns=['k', 'retriever', 'task', 'target_metric']), a1], axis=1)
        single_df_to_write.to_csv(single_output_path, mode='a', index=False, header=not single_csvfile.exists())
        

    
        print('~learning linear regression')
    
        output_content = []
    
        all_ablations = ['p0', 'p1', 'p2', 'p3', 'p01', 'p02', 'p03', 'p12', 'p13', 'p23', 'p012', 'p013', 'p023', 'p123', 'p0123']
    
        # define features
        for use_postgen, pregen_combination in itertools.product([False, True], all_ablations):
        
            # print(pregen_combination, use_postgen)
            used_qpp_methods = ['nqc', 'spatial', 'maxScore', 'a_ratio', 'bertQPP']
    
            used_predictors_unchanged, used_predictors_need_log = [], []
            if('0' in pregen_combination):
                used_predictors_need_log += used_qpp_methods
            if('1' in pregen_combination):
                used_predictors_unchanged += ['max(perpC)', 'avg(perpC)', 'itg(perpC)']
            if('2' in pregen_combination):
                used_predictors_need_log += ['max(docQual)', 'avg(docQual)', 'itg(docQual)']
            if('3' in pregen_combination):
                used_predictors_unchanged += readability_cols

            if((used_predictors_unchanged+used_predictors_need_log)==[]):
                continue
    
            # load data
            dev_data_0, dev_data_1, test_data_0, test_data_1 = 0, 0, 0, 0
            if(used_predictors_unchanged != []):
                dev_data_0 = dev_res[used_predictors_unchanged]
                test_data_0 = test_res[used_predictors_unchanged]
            if(used_predictors_need_log != []):
                dev_data_1 = dev_res[used_predictors_need_log].apply(lambda x: np.log(1+x))
                test_data_1 = test_res[used_predictors_need_log].apply(lambda x: np.log(1+x))
    
            if(used_predictors_unchanged == []):
                dev_data, test_data = dev_data_1, test_data_1
            elif(used_predictors_need_log == []):
                dev_data, test_data = dev_data_0, test_data_0
            else:
                dev_data, test_data = np.hstack((dev_data_0, dev_data_1)), np.hstack((test_data_0, test_data_1))
            
            if(use_postgen):
                dev_data = np.hstack((dev_data, dev_res[['prob(k)']].values))
                test_data = np.hstack((test_data, test_res[['prob(k)']].values))
    
            # best before combination
            best_row = a1[a1.QPP_Method.isin(used_predictors_unchanged+used_predictors_need_log+use_postgen*['prob(k)'])].query('Spearman==Spearman.max()').iloc[0]
            # print(f'Best before combination is {best_row.QPP_Method}, rho={best_row.Spearman}, tau={best_row.Kendall}')
        
            # linear regression
            reg = linear_model.LinearRegression()
            reg.fit(dev_data, dev_res[target_metric].values)
            coefs = reg.coef_
            intercept = reg.intercept_
            predictions = (dev_data * coefs).sum(axis=1) + intercept
        
            feature_names = used_predictors_unchanged + used_predictors_need_log + (['prob(k)'] if use_postgen else [])
            model_key = f'{_k}-{_ret}-{_task}-{target_metric}-{pregen_combination}-post{int(use_postgen)}'
            model_record = {'features': feature_names, 'log1p_features': used_predictors_need_log,
                            'coefficients': coefs.tolist(), 'intercept': float(intercept)}
            (args.output_dir / 'models' / f'{model_key}.json').write_text(json.dumps(model_record, indent=2))

            predictions_test = ((test_data * coefs).sum(axis=1) + intercept)
    
            result_rho, result_tau = stats.spearmanr(predictions_test, test_res[target_metric])[0], stats.kendalltau(predictions_test, test_res[target_metric])[0]
            # print(f'Accuracy on Test set, rho={result_rho}, tau={result_tau}')
            output_content.append(['GPP' if target_metric=='f1' else 'RPP', _task, _ret, _k, use_postgen, str(pregen_combination), result_rho, result_tau, best_row.QPP_Method, best_row.Spearman, best_row.Kendall, test_data.shape[0]])
    
        temp_output = pd.DataFrame(output_content, columns=['Prediction Name', 'QA Task', 'Retriever', 'Top-Retrieved Docs', 'Use_Postgen', 'Combination_Number', 'Rho', 'Tau', 'Best Single Signal', 'Best Single Rho', 'Best Single Tau', 'Number of Queries'])
        csvfile = pathlib.Path(output_path)
        temp_output.to_csv(output_path, mode='a', index=False, header=not csvfile.exists())
        # deduplicate
        x = pd.read_csv(output_path)
        x = x.drop_duplicates()
        x.to_csv(output_path, index=False)


def main():
    args = parse_args()
    paths = required_inputs(args)
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print(f'Missing {len(missing)} of {len(paths)} required input files:')
        for p in missing:
            print(p)
        return 1
    if args.check_inputs:
        print(f'All {len(paths)} required input files exist; schemas not validated.')
        return 0
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit('Output directory must be new or empty to avoid duplicate or overwritten results.')
    (args.output_dir / 'heatmaps').mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'models').mkdir()
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
