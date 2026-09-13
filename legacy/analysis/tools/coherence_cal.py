from tools.matrix_tools import *
import pandas as pd
import numpy as np
import pickle as pkl
from sentence_splitter import SentenceSplitter
from memory_profiler import profile
from tqdm import tqdm
from collections import defaultdict

# enable @profile to print memory consumption
# @profile
def cal_coherence(_matrix, _k, _doc_length_dict, _window=0, _step=1, bidirectional=0, distribution='uniform'): # 0 for document average, no overlaping # bidirectional: 0: upper; 1: lower; 2:bidirectional
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

        points = np.array(range(len(_sub_cohs)))
        if(distribution=='uniform'):
            weights = np.array([1.0]*len(points))
        elif(distribution=='top-heavy'):
            weights = np.exp(-points)
        elif(distribution=='tail-heavy'):
            weights = np.exp(points+1-len(points))
        elif(distribution=='lost-in-the-middle'):
            weights = np.exp((2*points/(len(points)-1)-1)**2)
        
        if(len(_sub_cohs) > 0):
            _coh_dict.update({str(qid): np.sum(_sub_cohs*weights)/np.sum(weights)})

    return _coh_dict

# @profile
def get_res_and_dicts(_task, _ret):
    material_path = '../../rag_utility'
    splitter = SentenceSplitter(language='en')

    print('loading doc dict')
    if(_task == 'dl'):
        f = open(f'{material_path}/doc_dicts/msmarco_passage_dict.pkl', 'rb')
        dl_19_res = pd.read_csv(f'{material_path}/res/{_ret}_dl_19.csv')
        dl_20_res = pd.read_csv(f'{material_path}/res/{_ret}_dl_20.csv')
        res = pd.concat([dl_19_res, dl_20_res])
        res.qid = res.qid.astype('str')
    elif(_task == 'dev_small'):
        f = open(f'{material_path}/doc_dicts/msmarco_passage_dict.pkl', 'rb')
        res = pd.read_csv(f'{material_path}/res/{_ret}_dev_small.csv')
        res.qid = res.qid.astype('str')
    elif(_task[:3] == 'nq_'):
        f = open(f'{material_path}/doc_dicts/nq_wiki_dict.pkl', 'rb')
        res = pd.read_csv(f'{material_path}/res/{_ret}_{_task}.csv')
    elif(_task[:9] == 'hotpotqa_'):
        f = open(f'{material_path}/doc_dicts/hotpotqa_wiki_dict.pkl', 'rb')
        res = pd.read_csv(f'{material_path}/res/{_ret}_{_task}.csv')
    doc_dict = pkl.load(f)
    f.close()

    print('calculating document lengths')
    # doc_length_dict = {}
    # for qid in tqdm(res.qid.unique()):
    
    #     length_list = []
    #     doc_texts = res[(res.qid == qid)&(res['rank']<12)].docno.apply(lambda x: doc_dict[str(x)]).values
    #     sentences = []
    #     for doc_text in doc_texts:
    #         length_list.append(len(splitter.split(doc_text)))
    #     doc_length_dict.update({qid: length_list})

    calculation_range = 12
    cache_file_name = f'../analysis/tools/cache/split_dict_{_task}_{_ret}_{calculation_range}.pkl'
    try:
        print('~Loading it from the cache') 
        f = open(cache_file_name, 'rb')
        doc_length_dict = pkl.load(f)
        f.close()
    except:
        print('~Cannot find it in the cache') 
        doc_length_dict = defaultdict(list)
    
        top_docs = res[res['rank'] < calculation_range].copy()
        top_docs['doc_text'] = top_docs['docno'].astype(str).map(doc_dict)
    
        # Split documents and get lengths
        doc_length_dict = defaultdict(list)
        for qid, group in tqdm(top_docs.groupby('qid')):
            lengths = [len(splitter.split(text)) for text in group['doc_text']]
            doc_length_dict[qid] = lengths

        print('~Saving it to the cache') 
        f = open(cache_file_name, 'wb+')
        pkl.dump(doc_length_dict, f)
        f.close()
    
    return res, doc_dict, doc_length_dict

    
