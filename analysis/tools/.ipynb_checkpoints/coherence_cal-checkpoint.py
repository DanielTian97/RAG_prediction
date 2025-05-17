from tools.matrix_tools import *
import pandas as pd
import pickle as pkl
from sentence_splitter import SentenceSplitter


def cal_coherence(_matrix, _k, _doc_length_dict, _window=0, _step=1, bidirectional=0): # 0 for document average, no overlaping # bidirectional: 0: upper; 1: lower; 2:bidirectional
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
                    _sub_cohs.append(cal_column_avg_upper_triangle(submat, bidirectional))
                _start += _d_start
    
        else:
            ## fixed length average            
            _real_window = min(stc_num_in_top_k, _window)
            if(_real_window == 1):
                continue
            for _start in range(0, stc_num_in_top_k-_real_window+1, _step):
                submat = cut_mtx[_start:_start+_real_window,_start:_start+_real_window]
                _sub_cohs.append(cal_column_avg_upper_triangle(submat, bidirectional))
            
        if(len(_sub_cohs) > 0):
            _coh_dict.update({str(qid): np.mean(_sub_cohs)})

    return _coh_dict

def get_res_and_dicts(_task, _ret):
    material_path = '../../rag_utility'
    splitter = SentenceSplitter(language='en')

    if(_task == 'dl'):
        f = open(f'{material_path}/doc_dicts/msmarco_passage_dict.pkl', 'rb')
        dl_19_res = pd.read_csv(f'{material_path}/res/{_ret}_dl_19.csv')
        dl_20_res = pd.read_csv(f'{material_path}/res/{_ret}_dl_20.csv')
        res = pd.concat([dl_19_res, dl_20_res])
        res.qid = res.qid.astype('str')
    elif(_task == 'nq_test'):
        f = open(f'{material_path}/doc_dicts/nq_wiki_dict.pkl', 'rb')
        res = pd.read_csv(f'{material_path}/res/{_ret}_nq_test.csv')
    doc_dict = pkl.load(f)
    f.close()

    doc_length_dict = {}
    for qid in res.qid.unique():
    
        length_list = []
        doc_texts = res[(res.qid == qid)&(res['rank']<12)].docno.apply(lambda x: doc_dict[str(x)]).values
        sentences = []
        for doc_text in doc_texts:
            length_list.append(len(splitter.split(doc_text)))
        doc_length_dict.update({qid: length_list})

    return res, doc_dict, doc_length_dict

    
