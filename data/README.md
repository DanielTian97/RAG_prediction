# Local data inputs

This repository does **not** distribute the full experiment data, retrieved-document text, model checkpoints, or large precomputed feature files used in the ECIR 2026 experiments. The `data/` directory is primarily a documented location for users who want to reproduce the pipeline with their own local copies.

One deliberately small historical fixture is retained under `data/examples/`: `e5_k3_qpp_dl_smoke.csv`. It contains 10 DL queries from the original E5, cutoff-3 QPP output and only the `nqc` and `maxScore` rows needed to exercise the precomputed-QPP path. It is a smoke-test fixture, **not** a complete paper result file. The old collections of per-retriever/per-cutoff result files are intentionally not carried into the refactored branch.

A convenient layout for local reproduction data is:

```text
data/
├── examples/               # one compact committed smoke fixture
├── retrieval/
├── generations/
├── evaluations/
├── qpp/
└── doc_dicts/
```

The command-line scripts do not require these exact subdirectory names; explicit paths can be supplied. The layout above is simply the recommended convention.

## Retrieval results

`compute_features.py` requires a retrieval CSV for each dataset split / retriever condition. Each CSV must contain at least:

```text
qid, query, docno, rank, score
```

Example:

```text
qid,query,docno,rank,score
42,"who wrote ...",123456,0,18.72
42,"who wrote ...",812345,1,17.94
```

The retrieval file determines which passages are used by QPP, PerpC, QualT5 and readability.

## Generation results

The post-generation feature (`prob(k)`) is extracted from the RAG generation JSON passed through `--generation-json`.

For each query, the refactored code expects the same relevant structure as the original experiments:

```python
generations[qid]["0"]["0"]["probs"]
```

`probs` may be either a Python/JSON list of token probabilities or a string representation of such a list.

## Evaluation results

`analyse.py` requires evaluation JSON files for both:

- zero-context generation (`P_0`), and
- retrieved-context generation (`P_k`).

They are used to construct:

```text
GPP target = P_k
RPP target = P_k - P_0
```

For the retained ECIR setup, the relevant answer-quality value is F1. The script currently reads the original NQ and DL evaluation structures used by the experiments.

## Optional precomputed QPP files

Exact reproduction of the historical learned/dense QPP signals can use `--qpp-precomputed`.

The CSV must contain at least:

```text
qid, qpp_method, qpp_estimate
```

and may additionally contain `query`.

For example:

```text
qid,query,qpp_method,qpp_estimate
42,"who wrote ...",bertQPP,0.417
42,"who wrote ...",spatial,13.284
42,"who wrote ...",a_ratio,1.118
```

These files are optional because score-based signals such as MaxScore and NQC can be computed by the refactored pipeline directly. They are needed only when reproducing historical QPP methods for which this repository does not contain a complete standalone inference implementation.

The committed `data/examples/e5_k3_qpp_dl_smoke.csv` is intentionally much smaller than an experiment input: it contains 10 queries and two methods (`nqc`, `maxScore`) sampled from the historical E5-at-3 DL file. `scripts/smoke_pipeline.sh` uses it to check that historical QPP values can be merged with newly computed feature families and passed into `analyse.py`.

## Document-text dictionaries (`.pkl`)

The original experiments frequently loaded passage text from pickled Python dictionaries rather than directly from a PyTerrier text loader. For the ECIR datasets, the historical files were conceptually:

```text
data/doc_dicts/msmarco_passage_dict.pkl
data/doc_dicts/nq_wiki_dict.pkl
```

These files are **not** included in the repository.

The pickle must contain a Python mapping from document identifier to document/passage text:

```python
{
    "<docno>": "<document or passage text>",
    ...
}
```

A minimal example is:

```python
{
    "123456": "William Shakespeare was an English playwright, poet and actor ...",
    "123457": "The Globe Theatre was associated with William Shakespeare ...",
    "812345": "Another retrieved passage ...",
}
```

The important requirements are:

- keys correspond to the `docno` values in the retrieval CSV;
- keys should be strings (the loader also normalises keys with `str(...)`);
- values are the raw passage/document text strings supplied to the feature calculators;
- every retrieved `docno` needed by the selected top-k passages must be present.

For example, a compatible dictionary can be created with:

```python
import pickle

passages = {
    "123456": "First passage text ...",
    "123457": "Second passage text ...",
}

with open("data/doc_dicts/nq_wiki_dict.pkl", "wb") as f:
    pickle.dump(passages, f)
```

and checked with:

```python
import pickle

with open("data/doc_dicts/nq_wiki_dict.pkl", "rb") as f:
    passages = pickle.load(f)

assert isinstance(passages, dict)
assert all(isinstance(str(docno), str) for docno in passages)
assert all(isinstance(text, str) for text in passages.values())
```

Pass the file to feature computation with:

```bash
python compute_features.py \
  --retrieval-csv data/retrieval/nq_test/e5.csv \
  --generation-json data/generations/nq_test/e5_k3.json \
  --doc-dict data/doc_dicts/nq_wiki_dict.pkl \
  --dataset nq_test \
  --retriever e5 \
  --k 3
```

When `--doc-dict` is supplied, the document text in this pickle is used for PerpC, QualT5 and readability. This is the preferred route for matching the exact text consumed by the original experiments. If it is omitted, `compute_features.py` can instead use the configured PyTerrier text source.

## Files that should not be placed here

Large retrieval indices and model checkpoints should normally remain external to the repository, including Terrier indices, dense-vector/HNSW indices, Llama checkpoints, QualT5 model weights, and raw MS MARCO/Wikipedia collections.

Historical `split_dict_*.pkl` files are also **not required**. They were sentence-count caches used by older coherence experiments and are not primary inputs to the five final feature families.
