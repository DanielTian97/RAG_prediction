import json
import numpy as np

class Performance_Loader:

    pfm_path = '../../rag-correlation'
    
    def load_0shot_performances(self):
        f = open(f'{self.pfm_path}/random_answers_0shot_5calls_0_0_bm25_dl_19_prompt1_eval.json')
        evals_0 = json.load(f)
        f.close()
        
        f = open(f'{self.pfm_path}/random_answers_0shot_5calls_0_0_bm25_dl_20_prompt1_eval.json')
        evals_0.update(json.load(f))
        f.close()
        
        evals_0_dict = {}
        for qid in evals_0:
            # print(qid)
            _scores = []
            for ans in evals_0[qid]['0'].values():
                # print(ans)
                answer_score = max(ans['qrel_2']['f1']['max'], ans['qrel_3']['f1']['max'])
                _scores.append(answer_score)
            evals_0_dict.update({qid: np.mean(_scores)})
        return evals_0_dict
    
    # separated context
    def load_sep_performances(self, _ret, _k):
        if(_ret == 'bm25'):
            f = open(f'{self.pfm_path}/random_answers_{_k}shot_5calls_1_0_dl_19_prompt1_eval.json')
        else:
            f = open(f'{self.pfm_path}/random_answers_{_k}shot_5calls_1_0_{_ret}_dl_19_prompt1_eval.json')
        evals = json.load(f)
        f.close()
        
        if(_ret == 'bm25'):
            f = open(f'{self.pfm_path}/random_answers_{_k}shot_5calls_1_0_dl_20_prompt1_eval.json')
        else:
            f = open(f'{self.pfm_path}/random_answers_{_k}shot_5calls_1_0_{_ret}_dl_20_prompt1_eval.json')
        evals.update(json.load(f))
        f.close()
        
        evals_dict = {}
        for qid in evals:
            # print(qid)
            _scores = []
            try:
                for ans in evals[qid]['0'].values():
                    answer_score = max(ans['qrel_2']['f1']['max'], ans['qrel_3']['f1']['max'])
                    _scores.append(answer_score)
            except:
                continue # for the case that there are no such answers
            evals_dict.update({qid: np.mean(_scores)})
        return evals_dict
    
    # integrated context
    def load_itg_performances(self, _ret, _k):
        f = open(f'../../context_optimisation/evaluator/eval_results/eval_{_k}shot_5calls_{_ret}_dl_19_integrated_original.json')
        evals = json.load(f)
        f.close()
        
        f = open(f'../../context_optimisation/evaluator/eval_results/eval_{_k}shot_5calls_{_ret}_dl_20_integrated_original.json')
        evals.update(json.load(f))
        f.close()
        
        evals_dict = {}
        for qid in evals:
            # print(qid)
            _scores = []
            try:
                for ans in evals[qid].values():
                    answer_score = max(ans['qrel_2']['f1']['max'], ans['qrel_3']['f1']['max'])
                    _scores.append(answer_score)
            except:
                continue
            evals_dict.update({qid: np.mean(_scores)})
        return evals_dict