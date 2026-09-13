import pickle as pkl
import json
import pandas as pd
from load_performances import Performance_Loader
from ir_metric_tool import IR_Metrics
import numpy as np
from matrix_calculation_tools import *
from correlation_tool import *
from qpp_methods import *
import pyterrier as pt
import pyterrier_rag
import pyterrier_dr
from pyterrier_dr import E5
import math

class Main_Experiment():
    
    material_path = '../../rag_utility'

    f = open(f'{material_path}/doc_dicts/msmarco_passage_dict.pkl', 'rb')
    doc_dict = pkl.load(f)
    f.close()

    def load_supervised_qpp_res(self, _ret, _task='dl', _withQV=False):
        _ret_converter = {'mt5': 'bm25_monot5', 'e5': 'e5', 'bm25': 'bm25'}
        
        _suffix = 'matched' if _withQV else 'matched_withoutQV'
        _retriever = _ret_converter[_ret]
    
        _qpp_res_dict = {}

        if(_task in ['dl']):
            for _query_set in ['trec-dl-2019', 'trec-dl-2020']:
                with open(f'./supervised_results/QPP_bm25_bert-base-uncased_{_suffix}/results-{_query_set}-{_retriever}.txt') as f:
                    for l in f:
                        _qid, _qpp_value = l.rstrip().split('\t')
                        _qpp_res_dict.update({_qid: float(_qpp_value)})
                    f.close()
        else:
            with open(f'./supervised_results/QPP_bm25_bert-base-uncased_{_suffix}/results-{_task}-{_retriever}.txt') as f:
                for l in f:
                    _qid, _qpp_value = l.rstrip().split('\t')
                    _qpp_res_dict.update({_qid: float(_qpp_value)})
                f.close()
    
        # print(len(_qpp_res_dict)) # check the length of qppres dict
        return _qpp_res_dict 
        

    def experiment(self, _ret, _k, _task='dl'):

        # get res
        if(_task=='dl'):
            dl_19_res = pd.read_csv(f'{self.material_path}/res/{_ret}_dl_19.csv')
            dl_20_res = pd.read_csv(f'{self.material_path}/res/{_ret}_dl_20.csv')
            retr_res = pd.concat([dl_19_res, dl_20_res])
        else:
            retr_res = pd.read_csv(f'{self.material_path}/res/{_ret}_{_task}.csv')
            # patch for the different style of retrieval score of MonoT5 on datasets other than trec-dl
            if(_ret=='mt5'):
                retr_res['score'] = retr_res['score'].apply(lambda x: np.exp(x))
            
        qpp = QPP(task=_task)
            
        retr_res.qid = retr_res.qid.astype('str')
        query_df = retr_res[['qid', 'query']].drop_duplicates()

        e5_model = pyterrier_dr.E5()
        if(_task in ['dl', 'dev_small']):
            e5_index = pyterrier_dr.FlexIndex('../../get_res/e5_msmarco_index.flex')
        elif(_task in ['nq_test', 'nq_dev']):
            e5_index = pt.Artifact.from_hf('pyterrier/ragwiki-e5.flex')

        try:
            df_qpp = pd.read_csv(f'./precomputed_qpps/{_ret}_{_k}_combined_qpp_{_task}.csv')
        except:
            df_qpp = qpp.qpp_in_batch(retr_res, 'nqc', _k, e5_model, e5_index)
            df_qpp = pd.concat([df_qpp, qpp.qpp_in_batch(retr_res, 'maxScore', _k, e5_model, e5_index)])

        if(('spatial' not in df_qpp.qpp_method.unique())|('a_ratio' not in df_qpp.qpp_method.unique())):
            df_qpp = pd.concat([df_qpp, qpp.qpp_in_batch(retr_res, 'spatial', _k, e5_model, e5_index)])
            df_qpp = pd.concat([df_qpp, qpp.qpp_in_batch(retr_res, 'a_ratio', _k, e5_model, e5_index)])

        if('bertQPP' not in df_qpp.qpp_method.unique()):
            df_qpp_sup_noQV = query_df.copy()
            bertQPP_res = self.load_supervised_qpp_res(_ret, _task, False)
            df_qpp_sup_noQV['qpp_estimate'] = df_qpp_sup_noQV['qid'].apply(lambda x: bertQPP_res[x])
            df_qpp_sup_noQV['parameters'] = str({'encoder': 'bert-base-uncased'})
            df_qpp_sup_noQV['qpp_method'] = 'bertQPP'
            df_qpp = pd.concat([df_qpp, df_qpp_sup_noQV])

        if('bertQPP(QV)' not in df_qpp.qpp_method.unique()):
            df_qpp_sup_withQV = query_df.copy()
            bertQPP_qv_res = self.load_supervised_qpp_res(_ret, _task, True)
            df_qpp_sup_withQV['qpp_estimate'] = df_qpp_sup_withQV['qid'].apply(lambda x: bertQPP_qv_res[x])
            df_qpp_sup_withQV['parameters'] = str({'encoder': 'bert-base-uncased', 'k': 1})
            df_qpp_sup_withQV['qpp_method'] = 'bertQPP(QV)'
            df_qpp = pd.concat([df_qpp, df_qpp_sup_withQV])

        df_qpp.to_csv(f'./precomputed_qpps/{_ret}_{_k}_combined_qpp_{_task}.csv', index=False)
