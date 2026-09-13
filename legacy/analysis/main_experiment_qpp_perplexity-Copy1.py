import pickle as pkl
import json
import pandas as pd
from load_performances import Performance_Loader
from ir_metric_tool import IR_Metrics
import numpy as np
from matrix_calculation_tools import *
from correlation_tool import *
from sentence_splitter import SentenceSplitter
from qpp_methods import *
import pyterrier_dr
from pyterrier_dr import E5
import math

class Main_Experiment():
    
    material_path = '../../rag_utility'

    splitter = SentenceSplitter(language='en')

    f = open(f'{material_path}/doc_dicts/msmarco_passage_dict.pkl', 'rb')
    doc_dict = pkl.load(f)
    f.close()

    def normalise_dict_values(self, d):
        _min, _max = min(d.values()), max(d.values())
        norm_dict = {}
        for item in d.items():
            norm_dict.update({item[0]: (item[1]-_min)/(_max-_min)})
        print('normalised!')
        return norm_dict

    def load_supervised_qpp_res(self, _ret, _withQV=False):
        _ret_converter = {'mt5': 'bm25_monot5', 'e5': 'e5', 'bm25': 'bm25'}
        
        _suffix = 'matched' if _withQV else 'matched_withoutQV'
        _retriever = _ret_converter[_ret]
    
        _qpp_res_dict = {}
        
        for _query_set in ['trec-dl-2019', 'trec-dl-2020']:
            with open(f'./supervised_results/QPP_bm25_bert-base-uncased_{_suffix}/results-{_query_set}-{_retriever}.txt') as f:
                for l in f:
                    _qid, _qpp_value = l.rstrip().split('\t')
                    _qpp_res_dict.update({_qid: float(_qpp_value)})
                f.close()
    
        # print(len(_qpp_res_dict)) # check the length of qppres dict
        return _qpp_res_dict 
        

    def experiment(self, _ret, _k): #_w for window size

        # get res
        dl_19_res = pd.read_csv(f'{self.material_path}/res/{_ret}_dl_19.csv')
        dl_20_res = pd.read_csv(f'{self.material_path}/res/{_ret}_dl_20.csv')
        dl_res = pd.concat([dl_19_res, dl_20_res])
        dl_res.qid = dl_res.qid.astype('str')
    
        # prepare doc length dict
        doc_length_dict = {}
        for qid in dl_res.qid.unique():
            length_list = []
            doc_texts = dl_res[(dl_res.qid == qid)&(dl_res['rank']<50)].docno.apply(lambda x: self.doc_dict[str(x)]).values
            sentences = []
            for doc_text in doc_texts:
                length_list.append(len(self.splitter.split(doc_text)))
            doc_length_dict.update({qid: length_list})
    
        # load performances
        pfm_loader = Performance_Loader()
        
        # kshot_pfms = pfm_loader.load_itg_performances(_ret, _k)
        kshot_pfms = pfm_loader.load_sep_performances(_ret, _k)
        zeroshot_pfms = pfm_loader.load_0shot_performances()

        # load 0-shot posteriors
        with open('../posterior_process/res/dl_mean_posteriors.pkl', 'rb') as f:
            posteriors_0shot = pkl.load(f)
            f.close()
        
        utility_dict = {}
        for qid in set(kshot_pfms.keys()).intersection(set(zeroshot_pfms.keys())):
            utility_dict.update({qid: (kshot_pfms[qid] - zeroshot_pfms[qid])})

        # load perplexity
        f = open(f'../coherence_eval/log_prob_temp_res/full_context/dl_{_ret}_{_k}.json')
        k_probs = json.load(f)
        f.close()

        qpp = QPP()
        e5_model = pyterrier_dr.E5()
        e5_index = pyterrier_dr.FlexIndex('../../get_res/e5_msmarco_index.flex')

        df_qpp = qpp.qpp_in_batch(dl_res, 'spatial', _k, e5_model, e5_index)
        df_qpp = pd.concat([df_qpp, qpp.qpp_in_batch(dl_res, 'a_ratio', _k, e5_model, e5_index)])
        df_qpp = pd.concat([df_qpp, qpp.qpp_in_batch(dl_res, 'nqc', _k, e5_model, e5_index)])

        query_df_for_sup_qpp = dl_res[['qid', 'query']].drop_duplicates()
        
        df_qpp_sup_noQV = query_df_for_sup_qpp.copy()
        bertQPP_res = self.load_supervised_qpp_res(_ret, False)
        df_qpp_sup_noQV['qpp_estimate'] = df_qpp_sup_noQV['qid'].apply(lambda x: bertQPP_res[x])
        df_qpp_sup_noQV['parameters'] = str({'encoder': 'bert-base-uncased'})
        df_qpp_sup_noQV['qpp_method'] = 'bertQPP'
        df_qpp = pd.concat([df_qpp, df_qpp_sup_noQV])

        df_qpp_sup_withQV = query_df_for_sup_qpp.copy()
        bertQPP_qv_res = self.load_supervised_qpp_res(_ret, True)
        df_qpp_sup_withQV['qpp_estimate'] = df_qpp_sup_withQV['qid'].apply(lambda x: bertQPP_qv_res[x])
        df_qpp_sup_withQV['parameters'] = str({'encoder': 'bert-base-uncased', 'k': 1})
        df_qpp_sup_withQV['qpp_method'] = 'bertQPP(QV)'
        df_qpp = pd.concat([df_qpp, df_qpp_sup_withQV])

        #for utility
        result_content = []
        result_columns = ['retriever', 'k', 'qpp_method', 'alpha', 'r', 'p1(r)', 'tau', 'p1(tau)', 'rho', 'p1(rho)', 'num_of_queries', 'with_0shot_posterior']
    
        for _alpha in np.arange(0, 1.01, 0.05):  
            for _qpp_method in ['spatial', 'a_ratio', 'nqc', 'bertQPP']:
                test_qpp_df = df_qpp[df_qpp.qpp_method==_qpp_method]
                qpp_dict = dict(zip(test_qpp_df.qid.values, test_qpp_df.qpp_estimate.values))
                union_dict = {}
                union_dict_with_0pos = {}
                for qid in set(qpp_dict.keys()).intersection(set(k_probs.keys())):
                    union_dict.update({qid: (1-_alpha)*math.log(qpp_dict[qid])+_alpha*k_probs[qid]})
                    union_dict_with_0pos.update({qid: ((1-_alpha)*math.log(qpp_dict[qid])+_alpha*k_probs[qid])*(-posteriors_0shot[qid])})
                # union without posteriors
                [r, p1_r], [tau, p1_tau], [rho, p1_rho] = correlator_simple(utility_dict, union_dict, True)
                result_content.append([_ret, _k, _qpp_method, round(_alpha,2), r, p1_r, tau, p1_tau, rho, p1_rho, len(union_dict), False])
                # union with 0-shot posteriors
                [r, p1_r], [tau, p1_tau], [rho, p1_rho] = correlator_simple(utility_dict, union_dict_with_0pos, True)
                result_content.append([_ret, _k, _qpp_method, round(_alpha,2), r, p1_r, tau, p1_tau, rho, p1_rho, len(union_dict_with_0pos), True])
        # only posterior
        [r, p1_r], [tau, p1_tau], [rho, p1_rho] = correlator_simple(utility_dict, posteriors_0shot, True)
        result_content.append([_ret, _k, '0shot_pos', round(_alpha,2), r, p1_r, tau, p1_tau, rho, p1_rho, len(posteriors_0shot), True])
        
        result_df = pd.DataFrame(result_content, columns=result_columns)
        
        result_df.to_csv(f'./result_qpp_union/perplexity/{_ret}_{_k}.csv', index=False)

        #for answer quality
        result_content = []
        result_columns = ['retriever', 'k', 'qpp_method', 'alpha', 'r', 'p1(r)', 'tau', 'p1(tau)', 'rho', 'p1(rho)', 'num_of_queries', 'with_0shot_posterior']
    
        for _alpha in np.arange(0, 1.01, 0.05):  
            for _qpp_method in ['spatial', 'a_ratio', 'nqc', 'bertQPP']:
                test_qpp_df = df_qpp[df_qpp.qpp_method==_qpp_method]
                qpp_dict = dict(zip(test_qpp_df.qid.values, test_qpp_df.qpp_estimate.values))
                union_dict = {}
                union_dict_with_0pos = {}
                for qid in set(qpp_dict.keys()).intersection(set(k_probs.keys())):
                    union_dict.update({qid: (1-_alpha)*math.log(qpp_dict[qid])+_alpha*k_probs[qid]})
                    union_dict_with_0pos.update({qid: ((1-_alpha)*math.log(qpp_dict[qid])+_alpha*k_probs[qid])*(-posteriors_0shot[qid])})
                # union without posteriors
                [r, p1_r], [tau, p1_tau], [rho, p1_rho] = correlator_simple(kshot_pfms, union_dict, True)
                result_content.append([_ret, _k, _qpp_method, round(_alpha,2), r, p1_r, tau, p1_tau, rho, p1_rho, len(union_dict), False])
                # union with 0-shot posteriors
                [r, p1_r], [tau, p1_tau], [rho, p1_rho] = correlator_simple(kshot_pfms, union_dict_with_0pos, True)
                result_content.append([_ret, _k, _qpp_method, round(_alpha,2), r, p1_r, tau, p1_tau, rho, p1_rho, len(union_dict_with_0pos), True])
        # only posterior
        [r, p1_r], [tau, p1_tau], [rho, p1_rho] = correlator_simple(kshot_pfms, posteriors_0shot, True)
        result_content.append([_ret, _k, '0shot_pos', round(_alpha,2), r, p1_r, tau, p1_tau, rho, p1_rho, len(posteriors_0shot), True])
        
        result_df = pd.DataFrame(result_content, columns=result_columns)
        
        result_df.to_csv(f'./result_qpp_union/perplexity/{_ret}_{_k}_answer_quality.csv', index=False)
                    