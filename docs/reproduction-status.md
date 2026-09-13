# Reproduction status

This first-pass audit describes the current code, not a claim that the published results have been reproduced. Stored results remain unchanged. The final notebooks are now converted; see [notebook provenance](notebook-provenance.md) for exact changes and verification.

## Candidate workflow

| Stage | Code or artifact | Status |
| --- | --- | --- |
| Retrieval and generation | External `rag_utility` inputs | Complete upstream workflow not supplied here |
| QPP features | `analysis/qpp_methods.py`, `analysis/precomputed_qpps/` | Code and cached CSVs present; exact generating entry point needs tracing |
| Context log probabilities | `perplexity_eval/prob_estimator.py`, `log_prob_calculator*.py` | Requires retrieval runs, document dictionaries, and Llama access |
| Document quality | `qualt5_eval/quality_estimation.py`, `quality_estimation_integrated_context.py` | Scripts present; output CSVs ignored by Git |
| Readability | `readability_eval/human_readability*.ipynb`, `human_readability_individuals.py` | Requires passage text; output CSVs and pickle caches ignored |
| Regression | `analysis/run_experiments.py` | Reads external labels/features and writes single/ensemble summaries |
| Tables and plots | `analysis/report_results.py` | Reads `ecir_res/single_output_v2.csv` and `union_output_v2.csv`; exports numeric tables and figure data |
| Significance | `analysis/tools/fisher_test.py` | Helper retained; comparison with published results remains necessary |

`analysis/main_experiment.py`, `main_experiment_qpp.py`, and `load_performances.py` contain historical TREC DL/coherence workflows. Their names do not establish them as the NQ paper entry point.

## Required inputs and paths

Converted analysis scripts resolve paths independently of the working directory; `run_experiments.py --material-dir` selects external materials. The table below records the original notebook paths for provenance. Unconverted feature scripts generally still expect their component directory as the working directory. The authoritative upstream repository revision is not recorded here.

| Input | Expected location or structure |
| --- | --- |
| Retrieval CSVs | `../../rag_utility/res/{retriever}_{split}.csv`; consumers use `qid`, `query`, `docno`, `rank`, and retrieval scores for QPP |
| NQ passage dictionary | `../../rag_utility/doc_dicts/nq_wiki_dict.pkl`, addressed by string document IDs |
| Historical MS MARCO dictionary | `../../rag_utility/doc_dicts/msmarco_passage_dict.pkl` |
| NQ zero-shot labels | `../../rag_utility/eval_results/short_answers_0shot_1calls_0_0_bm25_dl_{split}_concise_eval.json` |
| NQ k-shot labels | Same directory, `short_answers_{k}shot_1calls_1_0_{retriever}_dl_{split}_concise_eval.json` |
| NQ generation | `../../rag_utility/gen_results/short_answers_{k}shot_1calls_1_0_{retriever}_dl_{split}_concise.json` |
| Context JSONs | `perplexity_eval/log_prob_temp_res/full_context_with_query/{split}_{retriever}_{k}.json` and `individual_with_query/{split}_{retriever}_20.json` |
| QPP features | `analysis/precomputed_qpps/{retriever}_{k}_combined_qpp_{split}.csv`; includes `qid`, `query`, `qpp_method`, `qpp_estimate` |
| Quality CSVs | `qualt5_eval/quality_res/{retriever}_{split}.csv` and `{retriever}_{split}_integrated_{k}.csv` |
| Readability CSVs | `readability_eval/readability_res/individual_readability_{split}_{retriever}_top_10.csv` and `integrated_readability_{split}_{retriever}_top_{k}.csv` |

NQ splits are `nq_dev` and `nq_test`; retriever aliases are `bm25`, `mt5`, and `e5`. Evaluation JSONs are indexed as `[qid]['0']['0']['F1']`. Generation JSONs supply `[qid]['0']['0']['probs']`, parsed from a string using `ast.literal_eval` in the converted script (originally `eval`). Confirm the stored values' meaning from the upstream generator.

QualT5/readability scripts select `pyterrier/ragwiki-terrier` for NQ. Historical paths include `/mnt/indices/msmarco-passage.terrier/` and, in older QPP code, `/mnt/indices/msmarco-passage.tct-hnp.flex`. Output directories must exist before scripts append CSVs or write JSONs.

## Environment inventory

Observed imports include pandas, NumPy, SciPy, scikit-learn, matplotlib, tqdm, PyTerrier, pyterrier_dr, sentence_splitter, memory_profiler, torch, transformers, pyterrier_quality, readability, nltk, and IPython. These are import names, not a tested installation recipe. Recover original versions before preparing an environment specification.

`qualt5_eval/requirements.txt` does not cover the whole repository and does not explicitly list the `pyterrier_quality` import used by the scripts. The readability script downloads NLTK `punkt_tab`. PyTerrier text access needs Java. The context estimator loads `meta-llama/Meta-Llama-3-8B-Instruct` using Transformers in float16; model access and sufficient memory are required. Confirm how this relates to the paper's quantised answer-generation model and recover that original environment.

## Original-notebook findings and current status

1. **Notebook state:** `final_analysis_v3.ipynb` assigns the RPP subset to `_df`, then uses `rpp_df` in `df_dict`. That variable is undefined in a fresh kernel. Resolved by the standalone report script; the original notebook is recoverable in Git history.
2. **Feature semantics:** `ProbEstimator` writes mean log probabilities. The regression notebook uses context values directly and computes `prob(k)` as the mean of stored `probs`, without visible exponentiation at that stage. The paper describes exponentiated mean log probabilities without negation. Trace upstream values and the publication revision before reconciling these representations.
3. **Transformations:** regression applies `log(1+x)` to QPP and QualT5 features, passes context/readability values through, and fits `LinearRegression()` with an intercept. Preserve these details during code extraction.
4. **Sample/feature selection:** readability sentinel values (`-1`), missing values, Spache availability, and common dev/test columns influence selected features and queries. Compare actual counts and feature lists with the publication before simplifying.
5. **Reruns:** the regression notebook appends single-predictor rows before its completed-experiment check, so reruns can duplicate them. Ensemble results are deduplicated separately. Future verification should use copied output paths.
6. **Weights:** `weights.json` keys omit feature combination and pre/post-generation setting, allowing later combinations to overwrite earlier ones. Only coefficients are stored. This is not a complete reusable model release.
7. **Artifact identity:** several notebook versions, summary versions, and QPP backups coexist. The v2 summaries now match all 84 numeric cells in Tables 1/2. Their experiment/presentation lineage and remaining provenance limits are recorded in notebook-provenance.md.

## Next milestone

Recover external NQ labels and feature outputs with their provenance, confirm the authoritative revision, and compare table cells and plotted values against the paper. A lightweight route can eventually refit predictors from complete cached features and labels. Aggregate summaries alone permit inspection and presentation, not independent validation of the underlying correlations.

## First-pass validation scope

Check retained Python source syntax without importing model/data-loading modules, validate notebook JSON and local documentation links, inspect CSV schemas, and verify that scientific source, notebooks, and data are byte-for-byte unchanged relative to the parent commit. These checks do not constitute GPU execution or reproduction of published metrics.

## Conversion update

The original rerun and weight-saving issues above are resolved operationally in `run_experiments.py`: outputs must be fresh, and model records include feature combination, pre/post setting, feature order, transforms, coefficients and intercept. The numerical transformations and sample selection are preserved. See `requirements-analysis.txt` for the analysis-only dependencies. Original-data training is still unverified; the controlled numerical parity test and stored-table checks are described in notebook-provenance.md.
