import pandas as pd
import numpy as np
import pathlib
import argparse
from tqdm import tqdm
import pyterrier as pt
from pyterrier_quality import QualT5

parser = argparse.ArgumentParser()
parser.add_argument("--res_name", type=str, default='e5_dev_small')
parser.add_argument("--k", type=int, default=3)

args = parser.parse_args()
res_name = args.res_name
_k = args.k
retr_res = pd.read_csv(f'../../rag_utility/res/{res_name}.csv')

if not pt.java.started():
    pt.java.init()

qt5 = QualT5('pyterrier-quality/qt5-small')

if('nq' in res_name):
    index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
else:
    index_path ='/mnt/indices/msmarco-passage.terrier/'
    index_ref = pt.IndexRef.of(index_path)
    index = pt.IndexFactory.of(index_ref)

text_loader = index.text_loader(["text"])

output_filename = f"./quality_res/{res_name}_integrated_{_k}.csv"
csvfile = pathlib.Path(output_filename)
if(csvfile.exists()):
    exist_output = pd.read_csv(output_filename)
    exist_qids = exist_output.qid.unique()
    del exist_output
else:
    exist_qids = []

df_content = []
for _qid in tqdm(retr_res.qid.unique()):
    if(_qid in exist_qids):
        continue
    csvfile = pathlib.Path(output_filename)
    integrated_text = ''
    for _t, _i in zip(text_loader(retr_res[(retr_res.qid==_qid)&(retr_res['rank']<_k)])['text'], range(_k)):
        integrated_text += f'Context {_i+1}: "{_t}";\n'
        
    df_content.append([_qid, retr_res[(retr_res.qid==_qid)]['query'].values[0], f'{_qid}_itg_{_k}', integrated_text])

    if((len(df_content)==10)|(_qid==retr_res.qid.unique()[-1])):
        df_for_esti = pd.DataFrame(df_content, columns=['qid', 'query', 'docno', 'text'])
        temp_output = qt5(df_for_esti)
        temp_output.to_csv(output_filename, mode='a', index=False, header=not csvfile.exists())
        df_content = []