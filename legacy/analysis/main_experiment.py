import pickle as pkl
import pandas as pd
from load_performances import Performance_Loader
from ir_metric_tool import IR_Metrics
import numpy as np
from matrix_calculation_tools import *
from correlation_tool import *
from sentence_splitter import SentenceSplitter

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

    def cal_coherence(self, _matrix, _k, _doc_length_dict, _window=0, _step=1): # 0 for document average, no overlaping
        _coh_dict = {}
        
        for _qid, mtx in _matrix.items():
            qid = str(_qid)
            stc_num_in_top_k = np.sum(_doc_length_dict[qid][:_k])
            cut_mtx = np.matrix(mtx)[:stc_num_in_top_k][:stc_num_in_top_k]
        
            _sub_cohs = [] # coherence in a sub-piece of context
    
            if(_window == 0):
                ## document average
                # print(_doc_length_dict[qid])
                _start = 0
                for _d_start in _doc_length_dict[qid][:_k]:
                    if(_d_start > 1):
                        submat = cut_mtx[_start:_start+_d_start,_start:_start+_d_start]
                        _sub_cohs.append(cal_column_avg_upper_triangle(submat))
                    _start += _d_start
    
            else:
                ## fixed length average            
                _real_window = min(stc_num_in_top_k, _window)
                if(_real_window == 1):
                    continue
                for _start in range(0, stc_num_in_top_k-_real_window+1, _step):
                    submat = cut_mtx[_start:_start+_real_window,_start:_start+_real_window]
                    _sub_cohs.append(cal_column_avg_upper_triangle(submat))
            
            if(len(_sub_cohs) > 0):
                # _coh_dict.update({str(qid): _sub_cohs[-1]})
                # print(qid, _sub_cohs)
                _coh_dict.update({str(qid): np.mean(_sub_cohs)})
                # print(qid, _doc_length_dict[qid][:_k], np.mean(_sub_cohs))
        return _coh_dict

    def experiment(self, _ret, _k, _w): #_w for window size

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
    
        # get ir metric dicts
        dl_ir_metrics = IR_Metrics()
        ir_metric_dicts = dl_ir_metrics.get_ir_metric_dict(dl_res, 0, _k)
    
        # load performances
        pfm_loader = Performance_Loader()
        
        # kshot_pfms = pfm_loader.load_itg_performances(_ret, _k)
        kshot_pfms = pfm_loader.load_sep_performances(_ret, _k)
        zeroshot_pfms = pfm_loader.load_0shot_performances()
        
        utility_dict = {}
        for qid in set(kshot_pfms.keys()).intersection(set(zeroshot_pfms.keys())):
            # utility_dict.update({qid: (kshot_pfms[qid] - zeroshot_pfms[qid])/zeroshot_pfms[qid]})
            utility_dict.update({qid: (kshot_pfms[qid] - zeroshot_pfms[qid])})
    
        # load coherence matrix
        f = open(f'../coherence_res/dl_20_{_ret}.pkl', 'rb') # currently we calculates the sentence-pair coherence to 20
        matrix = pkl.load(f)
        f.close()
    
        import math
        # _w = 5
        _s = math.ceil(_w/2) # it is the convention, take half of the window size as the step length
        
        coh_dict = self.cal_coherence(matrix, _k, _doc_length_dict=doc_length_dict, _window=_w, _step=_s)
        union_dict = {}
        
        relevance_dict = ir_metric_dicts['ndcg']
        for qid in set(relevance_dict.keys()).intersection(set(coh_dict.keys())):
            # union_dict.update({qid: relevance_dict[qid]*coh_dict[qid]+coh_dict[qid]+0.5*relevance_dict[qid]})
            union_dict.update({qid: relevance_dict[qid]*coh_dict[qid]})
        correlator(utility_dict, union_dict, relevance_dict)
        correlator(utility_dict, relevance_dict, self.normalise_dict_values(coh_dict))
        # correlator(utility_dict, relevance_dict, coh_dict)