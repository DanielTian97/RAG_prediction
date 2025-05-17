from entail_tool import *
import pickle as pkl
import pandas as pd
import numpy as np
import numpy as np
from tqdm import tqdm
import math
import argparse

def get_sentence_splitter():
    from sentence_splitter import SentenceSplitter
    _splitter = SentenceSplitter(language='en')
    return _splitter

def save_results(content_to_save, save_to_path):
    f = open(save_to_path, 'wb')
    pkl.dump(full_coherence_dict, f)
    f.close()

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
    _output_path = f'../coherence_res/bi-directional/{_task}_{_k}_{_ret}.pkl'
    
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
        
    if(_task=='dl'):
        res = pd.concat([dl_19_res, dl_20_res])
    elif(_task=='nq_test'):
        res = pd.read_csv(f'{material_path}/res/{_ret}_nq_test.csv')

    try:
        f = open(_output_path, 'rb')
        full_coherence_dict = pkl.load(f)
        f.close()
    except:
        full_coherence_dict = {}

    for qid in tqdm(res.qid.unique()):
        if(qid in full_coherence_dict.keys()):
            continue

        doc_texts = res[(res.qid == qid) & (res['rank'] <_k)].docno.apply(lambda x: doc_dict[str(x)]).values
        sentences = []
        for doc_text in doc_texts:
            sentences += splitter.split(doc_text)

        full_entail_matrix = np.array(len(sentences)*[[-1.0]*len(sentences)])
        calculation_queue = []
        max_w = 16
        for i in range(len(sentences)-1): # the last one don't need to be calculated
            for j in range(i+1, i+max_w+1): # only consider the sentences after it; consider the maximum window size we explore: 16
                if(j == len(sentences)):
                    break
                calculation_queue.append((i, j))
                calculation_queue.append((j, i))

        # calculations of entailment
        batch_length = 20
        for i in range(math.ceil(len(calculation_queue)/batch_length)):
            _in_this_batch = calculation_queue[i*batch_length: min((i+1)*batch_length, len(calculation_queue))]
            nli_inputs = [[sentences[j[0]] for j in _in_this_batch], [sentences[j[1]] for j in _in_this_batch]]
            
            nli_outputs = judger.get_entailment(nli_inputs[0], nli_inputs[1])
            for position, value in zip(_in_this_batch, nli_outputs):
                full_entail_matrix[position] = value
            torch.cuda.empty_cache()
        
        full_coherence_dict.update({qid: full_entail_matrix})

        save_results(full_coherence_dict, _output_path)