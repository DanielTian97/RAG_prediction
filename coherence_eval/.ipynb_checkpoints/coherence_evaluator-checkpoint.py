from entail_tool import *
import pickle as pkl
import pandas as pd
import numpy as np
import numpy as np
from tqdm import tqdm
import argparse

def get_sentence_splitter():
    from sentence_splitter import SentenceSplitter
    _splitter = SentenceSplitter(language='en')
    return _splitter

if __name__=="__main__":
      
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, default='mt5')
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--task", type=str, default='dl')
    args = parser.parse_args()

    material_path = '../../rag_utility'
    
    _ret = args.retriever
    _k = args.k
    _task = args.task
    
    judger = EntailmentDeberta()

    # load msmarco passage dict
    if(_task == 'dl'):
        f = open(f'{material_path}/doc_dicts/msmarco_passage_dict.pkl', 'rb')
    elif(_task == 'nq_test'):
        f = open(f'{material_path}/doc_dicts/nq_wiki_dict.pkl', 'rb')
    doc_dict = pkl.load(f)
    f.close()

    # load res file
    if(_task=='dl'):
        dl_19_res = pd.read_csv(f'{material_path}/res/{_ret}_dl_19.csv')
        dl_20_res = pd.read_csv(f'{material_path}/res/{_ret}_dl_20.csv')
        res = pd.concat([dl_19_res, dl_20_res])
    elif(_task=='nq_test'):
        res = pd.read_csv(f'{material_path}/res/{_ret}_nq_test.csv')

    splitter = get_sentence_splitter()

    # doc_length_dict = {}
    # for qid in res.qid.unique():
    #     total_sentences = 0
    #     length_list = []
    #     doc_texts = res[(res.qid == qid) & (res['rank'] < _k)].docno.apply(lambda x: doc_dict[str(x)]).values
    #     sentences = []
    #     for doc_text in doc_texts:
    #         length_list.append(len(splitter.split(doc_text)))
    #         total_sentences += len(splitter.split(doc_text))
    #     doc_length_dict.update({qid: length_list})
        
    if(_task=='dl'):
        res = pd.concat([dl_19_res, dl_20_res])
    elif(_task=='nq_test'):
        res = pd.read_csv(f'{material_path}/res/{_ret}_nq_test.csv')


    full_coherence_dict = {}

    for qid in tqdm(res.qid.unique()):
        
        doc_texts = res[(res.qid == qid) & (res['rank'] <_k)].docno.apply(lambda x: doc_dict[str(x)]).values
        sentences = []
        for doc_text in doc_texts:
            sentences += splitter.split(doc_text)

        full_entail_matrix = []
        for i in range(len(sentences)-1): # the last one don't need to be calculated
            batch_length = min(16, len(sentences)-i-1)   # only consider the sentences after it; consider the maximum window size we explore: 16
            x = judger.get_entailment(sentences[i+1:i+1+batch_length], (batch_length)*[sentences[i]])
            x = (i+1)*[-1] + x + (len(sentences)-i-1-len(x))*[-1]
            full_entail_matrix.append(x)
            torch.cuda.empty_cache()
        full_entail_matrix.append(len(sentences)*[-1])
        
        full_coherence_dict.update({qid: full_entail_matrix})

    f = open(f'../coherence_res/{_task}_{_k}_{_ret}.pkl', 'wb')
    pkl.dump(full_coherence_dict, f)
    f.close()