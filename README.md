# Predicting Retrieval Utility and Answer Quality in RAG

Research code for **Predicting Retrieval Utility and Answer Quality in Retrieval-Augmented Generation**, by Fangzheng Tian, Debasis Ganguly, and Craig Macdonald (ECIR 2026).

[Paper](https://arxiv.org/abs/2601.14546) · [Reproduction status](docs/reproduction-status.md)

## Prediction tasks

- **Retrieval Performance Prediction (RPP):** predict context utility, the difference between answer quality with and without retrieved context: `U = P_k - P_0`.
- **Generation Performance Prediction (GPP):** predict answer quality with retrieved context: `P_k`.

The paper combines QPP signals, context perplexity (PerpC), document quality and readability, and post-generation answer perplexity (PerpA) through linear regression. NQ experiments use BM25, BM25 followed by MonoT5, and E5, with context sizes 2, 3, 5, 7, and 10. Prediction accuracy is evaluated using Spearman correlation.

## Current status

This experimental codebase is undergoing cleanup. Cached QPP features and result summaries are included, but **the repository is not yet a self-contained reproduction package**. Several inputs come from external generation/evaluation directories. Historical TREC DL and coherence experiments are also retained.

All experiment source files, notebooks, precomputed features, and results are preserved in this first cleanup. Read the [reproduction audit](docs/reproduction-status.md) before running notebooks: some append to existing outputs, and the final presentation notebook has a known fresh-kernel issue.

## Directory guide

| Location | Contents |
| --- | --- |
| `analysis/` | QPP methods, correlation utilities, exploratory notebooks, and regression analyses |
| `analysis/precomputed_qpps/` | Cached QPP feature CSVs by retriever, context size, and split |
| `analysis/ecir_res/` | Single-predictor and ensemble result summaries, including earlier versions |
| `analysis/plotting/` | Plotting notebook and stored figures |
| `analysis/supervised_results/` | Supervised QPP outputs and model/tokenizer metadata |
| `perplexity_eval/` | Individual/concatenated and query-conditioned context log-probability estimation |
| `qualt5_eval/` | Individual-document and concatenated-context QualT5 scoring |
| `readability_eval/` | Document and context readability computation |
| `posterior_process/` | Historical zero-shot posterior processing |
| `docs/` | Reproduction audit and outstanding requirements |

## Getting started

Clone the repository, then inspect stored summaries without loading models:

```bash
git clone https://github.com/DanielTian97/RAG_prediction.git
cd RAG_prediction
python - <<'PY'
import csv
from pathlib import Path
for path in sorted(Path("analysis/ecir_res").glob("*.csv")):
    with path.open(newline="") as handle:
        rows = csv.reader(handle)
        columns = next(rows)
        count = sum(1 for _ in rows)
    print(path.name, count, "rows", columns)
PY
```

This standard-library example reads existing outputs; it does not rerun experiments. There is no validated repository-wide installation command yet. `qualt5_eval/requirements.txt` is component-specific and incomplete for the full workflow. See the audit for observed dependencies and required inputs.

## Experiment navigation

`analysis/easy_analysis_readability-v2.ipynb` assembles features and targets, fits regressions, and writes the `*_output_v2.csv` summaries. `analysis/final_analysis_v3.ipynb` reads these summaries to prepare tables and context-size plots. These are candidate paper entry points based on their contents; exact reproduction of published numbers remains to be validated.

Code aliases include `mt5` for BM25 followed by MonoT5, `spatial` for DenseQPP, `a_ratio` for A-Pair-Ratio, and `prob(k)` for answer confidence. Probability transformations require reconciliation with the paper before refactoring, as explained in the audit.
