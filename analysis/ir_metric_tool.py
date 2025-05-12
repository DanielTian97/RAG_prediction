import pandas as pd
import ir_measures
from ir_measures import *
from tqdm import tqdm

class IR_Metrics:

    material_path = '../../rag_utility'
    
    def cal_ir_metric(self, runs_file_raw, target_qid, start, _k, qrels_df, col_metric='nDCG'):
        
        part_run_data = []
        for row in runs_file_raw[(runs_file_raw['rank'] >= start)&(runs_file_raw['rank'] < start+_k)].iterrows():
            values = row[1]
            qid = str(values.qid)
            docno = str(values.docno)
            score = values.score
            part_run_data.append({'query_id': qid, 'doc_id': docno, 'score': score})
        part_run = pd.DataFrame(part_run_data)
    
        for metric in ir_measures.iter_calc([nDCG@_k, AP@_k], qrels_df[qrels_df.query_id == target_qid], part_run):
            # print(metric)
            if(str(metric.measure) == f'{col_metric}@{_k}'):
                return metric.value
    
    def prepare_qrel_for_irmeasure(self):
        
        qrel_file_raw = pd.read_csv(f'{self.material_path}/qrels/qrels.csv')
        qrel_file_raw.qid = qrel_file_raw.qid.astype('str')
        data_for_ir_measure_qrels = []
        for row in qrel_file_raw.iterrows():
            values = row[1]
            qid = str(values.qid)
            docno = str(values.docno)
            label = values.label
            data_for_ir_measure_qrels.append({'query_id': qid, 'doc_id': docno, 'relevance': label})
        qrels_for_irmeasures = pd.DataFrame(data_for_ir_measure_qrels)
        return qrels_for_irmeasures
    
    def get_ir_metric_dict(self, res, start=0, _k=5):
        qrels = self.prepare_qrel_for_irmeasure()
        
        ndcg_dict = {}
        ap_dict = {}
        
        for qid in tqdm(res.qid.unique()):
            ndcg_dict.update({qid: self.cal_ir_metric(res, qid, 0, _k, qrels, 'nDCG')})
            ap_dict.update({qid: self.cal_ir_metric(res, qid, 0, _k, qrels, 'AP')})
        return {'ndcg': ndcg_dict, 'ap': ap_dict}