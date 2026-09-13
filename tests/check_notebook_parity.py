"""Compare converted numerical output with the original notebook on synthetic NQ data.
Requires Git history at the recorded source revision. Progress display is stubbed
only if tqdm is absent; all scientific libraries and input/output paths are real.
Run: python tests/check_notebook_parity.py
"""
import json,sys,tempfile,shutil,os,importlib.util,contextlib,io
from pathlib import Path
import numpy as np,pandas as pd
repo=Path(__file__).resolve().parents[1]
base=Path(tempfile.mkdtemp(prefix='ecir-parity-'));root=base/'RAG_prediction';material=base/'rag_utility'
(root/'analysis').mkdir(parents=True);shutil.copytree(repo/'analysis/tools',root/'analysis/tools')
sys.path.insert(0,str(root/'analysis'))
rng=np.random.default_rng(721)
for split in ['nq_dev','nq_test']:
 qids=[f'q{i}' for i in range(40)];ret='bm25';k=2
 def write(path,data):
  path.parent.mkdir(parents=True,exist_ok=True)
  if isinstance(data,pd.DataFrame):data.to_csv(path,index=False)
  else:path.write_text(json.dumps(data))
 for shots in [0,2]:
  stem=f'short_answers_{shots}shot_1calls_{int(shots>0)}_0_bm25_dl_{split}_concise'
  write(material/'eval_results'/f'{stem}_eval.json',{q:{'0':{'0':{'F1':float(rng.random())}}} for q in qids})
  if shots:write(material/'gen_results'/f'{stem}.json',{q:{'0':{'0':{'probs':str(rng.random(5).tolist())}}} for q in qids})
 write(root/f'perplexity_eval/log_prob_temp_res/full_context_with_query/{split}_bm25_2.json',{q:float(-rng.random()) for q in qids})
 write(root/f'perplexity_eval/log_prob_temp_res/individual_with_query/{split}_bm25_20.json',{q:{str(j):float(-rng.random()) for j in range(20)} for q in qids})
 write(root/f'analysis/precomputed_qpps/bm25_2_combined_qpp_{split}.csv',pd.DataFrame([{'qid':q,'query':q,'qpp_method':m,'qpp_estimate':rng.random()} for q in qids for m in ['nqc','spatial','maxScore','a_ratio','bertQPP','bertQPP(QV)']]))
 write(root/f'qualt5_eval/quality_res/bm25_{split}.csv',pd.DataFrame([{'qid':q,'docno':q+str(j),'rank':j,'quality':-rng.random()} for q in qids for j in range(2)]))
 write(root/f'qualt5_eval/quality_res/bm25_{split}_integrated_2.csv',pd.DataFrame([{'qid':q,'docno':q,'quality':-rng.random()} for q in qids]))
 write(root/f'readability_eval/readability_res/individual_readability_{split}_bm25_top_10.csv',pd.DataFrame([{'qid':q,'docno':q+str(j),'rank':j,'readability_metric':'Spache','readability_score':rng.random()+1} for q in qids for j in range(2)]))
 write(root/f'readability_eval/readability_res/integrated_readability_{split}_bm25_top_2.csv',pd.DataFrame([{'qid':q,'docno':q,'readability_metric':'Spache','score':rng.random()+1} for q in qids]))
for d in ['ecir_res','plotting/heatmaps','temp_for_pasting_results']:(root/'analysis'/d).mkdir(parents=True)
import subprocess
n=json.loads(subprocess.check_output(['git','show','68cc7b5c770954901df2d3eb75ab88aca69febb6:analysis/easy_analysis_readability-v2.ipynb'],cwd=repo));src='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
src=src.replace("[2, 3, 5, 7, 10], ['bm25', 'mt5', 'e5'], ['nq', 'dl']","[2], ['bm25'], ['nq']")
import types
try:
 import tqdm
except ImportError:
 shim=types.ModuleType('tqdm');shim.tqdm=lambda iterable, **kwargs: iterable;sys.modules['tqdm']=shim
import matplotlib;matplotlib.use('Agg')
old=Path.cwd();os.chdir(root/'analysis')
with contextlib.redirect_stdout(io.StringIO()):exec(compile(src,'original-notebook','exec'),{})
os.chdir(old)
spec=importlib.util.spec_from_file_location('converted',repo/'analysis/run_experiments.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.ROOT=root
from argparse import Namespace
out=base/'converted';(out/'models').mkdir(parents=True);(out/'heatmaps').mkdir()
args=Namespace(material_dir=material,output_dir=out,context_sizes=[2],retrievers=['bm25'],tasks=['nq'])
assert all(p.exists() for p in mod.required_inputs(args))
with contextlib.redirect_stdout(io.StringIO()):mod.run(args)
for filename in ['single_output_v2.csv','union_output_v2.csv']:
 a=pd.read_csv(root/'analysis/ecir_res'/filename);b=pd.read_csv(out/filename)
 pd.testing.assert_frame_equal(a,b,rtol=1e-12,atol=1e-12);print(filename,len(a),'rows match original notebook')
assert len(list((out/'models').glob('*.json')))==60
print('All 60 models saved independently. Synthetic NQ parity passed.')
