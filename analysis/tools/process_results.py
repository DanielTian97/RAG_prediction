import pandas as pd
import numpy as np
import json

def process_qualt5_res_individual(_path: str, _cutoff: int, _readability_metric=''):
    _quality_df = pd.read_csv(_path)
    if('readability' in _path):
        _quality_df = _quality_df.rename(columns={'readability_score': 'quality'})
        _quality_df = _quality_df.query('readability_metric==@_readability_metric')
    else:
        _quality_df.quality = _quality_df.quality.apply(lambda x: np.exp(x))
    qualt5_individual_res = _quality_df.query('rank<@_cutoff')[['qid', 'docno', 'rank', 'quality']]

    
    qualt5_individual_res = qualt5_individual_res[qualt5_individual_res.quality != -1]
    qualt5_individual_res_groupby_qid = qualt5_individual_res.groupby(['qid'])['quality']
    _doc_qual_max = dict(zip(qualt5_individual_res_groupby_qid.max().index.astype('str').values, qualt5_individual_res_groupby_qid.max().values))
    _doc_qual_avg = dict(zip(qualt5_individual_res_groupby_qid.mean().index.astype('str').values, qualt5_individual_res_groupby_qid.mean().values))
    _doc_qual_min = dict(zip(qualt5_individual_res_groupby_qid.max().index.astype('str').values, qualt5_individual_res_groupby_qid.min().values))
    return _doc_qual_max, _doc_qual_avg, _doc_qual_min

def process_qualt5_res_integrated(_path: str, _readability_metric=''):
    _quality_df = pd.read_csv(_path)
    if('readability' in _path):
        _quality_df = _quality_df.rename(columns={'score': 'quality'})
        _quality_df = _quality_df.query('readability_metric==@_readability_metric')
    else:
        _quality_df.quality = _quality_df.quality.apply(lambda x: np.exp(x))
        
    qualt5_integrated_res = _quality_df[['qid', 'docno', 'quality']]

    _doc_qual_itg = dict(zip(qualt5_integrated_res.qid.astype('str').values, qualt5_integrated_res.quality.values))
    return _doc_qual_itg

def tool_for_aggregating_dl_performance(x):
    # the same as performLoader
    scores = []
    for answer_eval in x[1]['0'].values():
        scores.append(max(answer_eval['qrel_2']['f1']['max'], answer_eval['qrel_3']['f1']['max']))
    return np.mean(scores)

def load_json(path):
    with open(path) as f:
        o = json.load(f)
    return o