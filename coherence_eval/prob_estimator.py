# CUDA_VISIBLE_DEVICES="0" python ... &
# CUDA_VISIBLE_DEVICES="1" python ...
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analysis')))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import torch.nn.functional as F
from analysis.tools import coherence_cal
import json
from tqdm import tqdm

class ProbEstimator:
    def __init__(self, _ret='bm25', _task='nq_test', model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.retriever = _ret
        self.task = _task
        self.res, self.doc_dict, _ = coherence_cal.get_res_and_dicts(_task, _ret)

        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model = self.model.to(self.device)

        self.MAX_BATCH_FOR_PASSAGES = 20

    def change_retriever(self, _new_ret):
        self.res, self.doc_dict, _ = coherence_cal.get_res_and_dicts(self.task, _new_ret)
    
    def cal_part_logprob_in_batch(self, _input_text: list, _device, _max_length=512):
        with torch.no_grad():
            inputs = self.tokenizer(_input_text, return_tensors="pt", padding=True, truncation=True, max_length=_max_length).to(_device)
            outputs = self.model(**inputs, labels=inputs["input_ids"], loss_type='ForCausalLMLoss')
        
            logits = outputs.logits.to('cpu')
            del outputs
            inputs = inputs.to('cpu')
            attention_mask = inputs["attention_mask"]
            input_ids = inputs["input_ids"]
            
            shift_logits = logits[: , :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            log_probs_for_tokens = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            actual_log_probs_for_tokens = attention_mask[:, 1:] * log_probs_for_tokens
            
            actual_nums = attention_mask[:, 1:].sum(dim=1)
            actual_sums = actual_log_probs_for_tokens.sum(dim=1)
            avg_logProbs = actual_sums/actual_nums
        return avg_logProbs.tolist()
    
    def individual_doc_probs(self, _k: int):
        output_path = f'./log_prob_temp_res/{self.task}_{self.retriever}_{_k}.json'
        
        try:
            f = open(output_path, 'r')
            prob_res = json.load(f)
            f.close()
        except:
            prob_res = {}
        
        for qid in tqdm(self.res.qid.unique()):
            if(qid in prob_res.keys()):
                continue
            _res_per_q = {}
        
            _batch_size = min(_k, self.MAX_BATCH_FOR_PASSAGES)
            _i = 0
            while _i < _k:
                doc_texts = self.res[(self.res.qid==qid)&(self.res['rank']>=_i)&(self.res['rank']<_i+_batch_size)].docno.apply(lambda x: self.doc_dict[str(x)]).tolist()
                _res_per_q = {}
                avg_logProb_list = self.cal_part_logprob_in_batch(doc_texts, self.device)
                torch.cuda.empty_cache()
                _res_per_q.update(dict(zip(range(_i, _i+len(doc_texts)), avg_logProb_list)))
                _i += _batch_size
        
            prob_res.update({qid: _res_per_q})
        
            f = open(output_path, 'w')
            json.dump(prob_res, f)
            f.close()    
        
    
    def concatenated_context_probs(self, _k: int):
        prob_res = {}
        output_path = f'./log_prob_temp_res/full_context/{self.task}_{self.retriever}_{_k}.json'
        
        try:
            f = open(output_path, 'r')
            prob_res = json.load(f)
            f.close()
        except:
            prob_res = {}

        _max_tokens = 1024 + 256*max(0, _k-5)
        _batch_size = max(1, math.floor(5120/_max_tokens))
        _i = 0
        doc_texts = []
        _qid_to_write = []
        for qid in tqdm(self.res.qid.unique()):
            if(qid in prob_res.keys()):
                continue
            doc_text = ''.join(self.res[(self.res.qid==qid)&(self.res['rank']<_k)].docno.apply(lambda x: self.doc_dict[str(x)]).tolist())
            doc_texts.append(doc_text)
            _qid_to_write.append(qid)
            _i += 1
            
            if ((_i == _batch_size)|(qid == self.res.qid.unique()[-1])):
                avg_logProb_list = self.cal_part_logprob_in_batch(doc_texts, self.device, _max_tokens)
                torch.cuda.empty_cache()
                prob_res.update(dict(zip(_qid_to_write, avg_logProb_list)))
                _qid_to_write = []
                doc_texts = []
                _i = 0
                
                f = open(output_path, 'w')
                json.dump(prob_res, f)
                f.close()