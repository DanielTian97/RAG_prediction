#!/usr/bin/env python3
# coding: utf-8

# Converted from human_readability_individuals.ipynb
# Markdown cells are preserved as comments.


# %% [cell 1: code]
from readability import Readability
import nltk

nltk.download('punkt_tab')

# %% [cell 2: code]
import pandas as pd

# %% [cell 3: code]
dataset = 'dev_small'
retriever = 'mt5'
retr_res = pd.read_csv(f'../../rag_utility/res/{retriever}_{dataset}.csv')

# %% [cell 4: code]
import pyterrier as pt

if not pt.java.started():
    pt.java.init()

# %% [cell 5: code]
if('nq' in dataset):
    index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
else:
    index_path ='/mnt/indices/msmarco-passage.terrier/'
    index_ref = pt.IndexRef.of(index_path)
    index = pt.IndexFactory.of(index_ref)

# index_path ='/mnt/indices/msmarco-passage.terrier/'
# index_ref = pt.IndexRef.of(index_path)
# index = pt.IndexFactory.of(index_ref)

# %% [cell 6: code]
text_loader = index.text_loader(["text"])

# %% [cell 7: code]


# %% [cell 8: code]


def calculate_readability(_input_text: str, _docno: str, _rank: int, _qid: str, _cache_dict):
    try:
        cached_result = _cache_dict[_docno]
    except:
        cached_result = {}
        _cache_dict.update({_docno: cached_result})
        
    r = Readability(_input_text)
    # Map names to the corresponding method calls (lambdas delay execution)
    metrics = {
        "Dale Chall": lambda: r.dale_chall(),
        "Spache": lambda: r.spache(),
        "Flesch-Kincaid": lambda: r.flesch_kincaid(),
        "Flesch": lambda: r.flesch(),
        "Gunning Fog": lambda: r.gunning_fog(),
        "Coleman Liau": lambda: r.coleman_liau(),
        "ARI": lambda: r.ari(),
        "Linsear Write": lambda: r.linsear_write(),
        "SMOG": lambda: r.smog(),
    }
    
    to_output = []
    for name, func in metrics.items():
        try:
            try:
                readability_result = cached_result[name]
            except:
                readability_result = func()
                cached_result.update({name: readability_result})
                
            score = readability_result.score
            try:
                grade_level = readability_result.grade_levels
            except:
                grade_level = []
            to_output.append([_qid, _docno, _rank, name, score, grade_level])
        except Exception as e:
            to_output.append([_qid, _docno, _rank, name, -1, []])

    return to_output, _cache_dict

# %% [cell 9: code]
from tqdm import tqdm
import pathlib
import pickle as pkl

try:
    exist_qids = pd.read_csv(f"./readability_res/individual_readability_{dataset}_{retriever}_top_10.csv").qid.unique().values
except:
    exist_qids = []

if('nq' in dataset):
    cache_name = 'nq_wiki'
elif('dl' in dataset):
    cache_name = 'msmarco'
elif('dev' in dataset):
    cache_name = 'msmarco'
    
try:
    with open(f'./readability_res/{cache_name}.pkl', 'rb') as f:
        cache_dict = pkl.load(f)
except:
    cache_dict = {}
    
for qid in tqdm(retr_res.qid.unique()):
    if(qid in exist_qids):
        continue

    input = retr_res[(retr_res.qid==qid)&(retr_res['rank']<10)]

    df_with_loaded_text = text_loader(input)
    for _t, docno, doc_rank in zip(df_with_loaded_text.text.values, df_with_loaded_text.docno.values, df_with_loaded_text['rank'].values):
        df_content, cache_dict = calculate_readability(_t, str(docno), doc_rank, qid, cache_dict)
        temp_df = pd.DataFrame(df_content, columns=['qid', 'docno', 'rank', 'readability_metric', 'readability_score', 'grade_level'])
        csvfile = pathlib.Path(f"./readability_res/individual_readability_{dataset}_{retriever}_top_10.csv")
        temp_df.to_csv(f"./readability_res/individual_readability_{dataset}_{retriever}_top_10.csv", mode='a', index=False, header=not csvfile.exists())

with open(f'./readability_res/{cache_name}.pkl', 'wb+') as f:
    pkl.dump(cache_dict, f)

# %% [cell 10: code]


# %% [cell 11: code]

