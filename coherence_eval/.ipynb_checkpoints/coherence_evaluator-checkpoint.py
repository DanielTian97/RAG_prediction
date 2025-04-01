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
    args = parser.parse_args()

    material_path = '../../rag_utility'
    
    _ret = args.retriever
    _k = args.k
    
    judger = EntailmentDeberta()

    # load msmarco passage dict
    f = open(f'{material_path}/doc_dicts/msmarco_passage_dict.pkl', 'rb')
    doc_dict = pkl.load(f)
    f.close()

    # load res file

    dl_19_res = pd.read_csv(f'{material_path}/res/{_ret}_dl_19.csv')
    dl_20_res = pd.read_csv(f'{material_path}/res/{_ret}_dl_20.csv')
    dl_res = pd.concat([dl_19_res, dl_20_res])

    splitter = get_sentence_splitter()

    doc_length_dict = {}
    for qid in dl_res.qid.unique():

        length_list = []
        doc_texts = dl_res[(dl_res.qid == qid) & (dl_res['rank'] < _k)].docno.apply(lambda x: doc_dict[str(x)]).values
        sentences = []
        for doc_text in doc_texts:
            length_list.append(len(splitter.split(doc_text)))
        doc_length_dict.update({qid: length_list})
        

    dl_res = pd.concat([dl_19_res, dl_20_res])


    full_coherence_dict = {}

    for qid in tqdm(dl_res.qid.unique()):
        
        doc_texts = dl_res[(dl_res.qid == qid) & (dl_res['rank'] <_k)].docno.apply(lambda x: doc_dict[str(x)]).values
        sentences = []
        for doc_text in doc_texts:
            sentences += splitter.split(doc_text)
        
        full_entail_matrix = []
        for i in range(len(sentences)):
            full_row = []
            for j in range(len(sentences)):
                _score = judger.get_entailment(sentences[j], sentences[i])
                full_row.append(_score)
            full_entail_matrix.append(full_row)
        
        full_coherence_dict.update({qid: full_entail_matrix})

    f = open(f'../coherence_res/dl_{_k}_{_ret}.pkl', 'wb')
    pkl.dump(full_coherence_dict, f)
    f.close()