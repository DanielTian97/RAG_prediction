import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
import pyterrier as pt
from pyterrier_quality import QualT5
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--res_name", type=str, default='e5_dev_small')

args = parser.parse_args()
res_name = args.res_name
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

csvfile = pathlib.Path(f"./quality_res/{res_name}.csv")
if(csvfile.exists()):
    exist_output = pd.read_csv(f"./quality_res/{res_name}.csv")
    exist_qids = exist_output.qid.unique()
    del exist_output
else:
    exist_qids = []

for _qid in tqdm(retr_res.qid.unique()):
    if(_qid in exist_qids):
        continue
    csvfile = pathlib.Path(f"./quality_res/{res_name}.csv")
    temp_output = qt5(text_loader(retr_res[retr_res.qid==_qid].head(20)))
    temp_output.to_csv(f"./quality_res/{res_name}.csv", mode='a', index=False, header=not csvfile.exists())