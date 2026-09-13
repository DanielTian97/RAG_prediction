# Predicting Retrieval Utility and Answer Quality in RAG

Research code for **Predicting Retrieval Utility and Answer Quality in Retrieval-Augmented Generation**, by Fangzheng Tian, Debasis Ganguly, and Craig Macdonald (ECIR 2026).

[Paper](https://arxiv.org/abs/2601.14546) · [Reproduction status](docs/reproduction-status.md)

## Prediction tasks

- **Retrieval Performance Prediction (RPP):** predict context utility, the difference between answer quality with and without retrieved context: `U = P_k - P_0`.
- **Generation Performance Prediction (GPP):** predict answer quality with retrieved context: `P_k`.

The paper combines QPP signals, context perplexity (PerpC), document quality and readability, and post-generation answer perplexity (PerpA) through linear regression. NQ experiments use BM25, BM25 followed by MonoT5, and E5, with context sizes 2, 3, 5, 7, and 10. Prediction accuracy is evaluated using Spearman correlation.

## Current status

This experimental codebase is undergoing cleanup. Cached QPP features and result summaries are included, but **the repository is not yet a self-contained reproduction package**. Several inputs come from external generation/evaluation directories. Historical TREC DL and coherence experiments are also retained.

The final experiment and presentation notebooks have been converted to Python, and their superseded versions removed. Stored results and precomputed features are unchanged. See [notebook provenance](docs/notebook-provenance.md) for the evidence, retained supporting notebooks, and recovery instructions.

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

This standard-library example reads existing outputs; it does not rerun experiments. `requirements-analysis.txt` covers the converted analysis scripts; feature extraction and generation require additional dependencies. See the audit for external inputs and environment limitations.

## Run the final analysis

Install the analysis dependencies (this is not the full feature-extraction environment):

```bash
python -m pip install -r requirements-analysis.txt
```

Export Tables 1/2 and Figure 3 from the included summaries:

```bash
python analysis/report_results.py --output-dir /tmp/ecir-report
```

Use a new or empty output directory. Exports include CSVs, plain LaTeX tables, a separate notebook-significance CSV, and PDF/SVG/PNG plots. This reconstructs presentation from stored correlations; it does not refit models. All 84 numeric cells in Tables 1/2 match the paper at four decimal places.

Check external inputs before rerunning the experiment:

```bash
python analysis/run_experiments.py --material-dir /path/to/rag_utility --output-dir /tmp/ecir-run --check-inputs
```

Once inputs are available, omit `--check-inputs` to run. NQ is the default; use `--context-sizes 2 --retrievers e5` for a smaller run, or `--tasks nq dl` to include the historical DL experiments. The script uses the existing feature directories in this checkout and requires a new or empty output directory. Paths are independent of the shell working directory.

Code aliases: `mt5` = BM25 followed by MonoT5, `spatial` = DenseQPP, `a_ratio` = A-Pair-Ratio, and `prob(k)` = answer confidence. Original probability transformations are preserved pending provenance reconciliation; see [reproduction status](docs/reproduction-status.md).

```bash
python -m unittest discover -s tests
python tests/check_notebook_parity.py
```
