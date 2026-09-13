# Final notebook identification and conversion

## Evidence

The repository's `analysis/ecir_res/single_output_v2.csv` reproduces all 30 numeric cells of Table 1 in [arXiv:2601.14546](https://arxiv.org/pdf/2601.14546). Together with `union_output_v2.csv`, it reproduces all 54 numeric cells of Table 2, including PerpA alone, after rounding to four decimal places. The presentation notebook selects the same four Figure 3 series, E5 retriever, two targets, and five context sizes.

This establishes the final **stored result and presentation lineage**. It does not prove that rerunning the experiment on unavailable original inputs will reproduce those CSVs. The experiment notebook is identified by its explicit write paths, full feature grid, and downstream consumer; its scientific provenance still has the caveats in the reproduction audit.

Source notebooks are recoverable at commit `68cc7b5c770954901df2d3eb75ab88aca69febb6`. For example:

```bash
git show 68cc7b5c770954901df2d3eb75ab88aca69febb6:analysis/easy_analysis_readability-v2.ipynb > /tmp/original-experiment.ipynb
```

## Final entry points

| Former notebook | Python replacement | Evidence and scope |
| --- | --- | --- |
| `analysis/easy_analysis_readability-v2.ipynb` | `analysis/run_experiments.py` | Writes both v2 summaries; loops over the five paper context sizes, all three retrievers, both targets, and pre/post-generation combinations |
| `analysis/final_analysis_v3.ipynb` | `analysis/report_results.py` | Reads those summaries; selects Table 1 QPP methods, Table 2 combinations p0/p01/p013/p0123, and Figure 3 E5 series |

The source notebooks have been replaced by scripts rather than retaining two editable copies. All stored results and precomputed features remain unchanged.

## Superseded notebooks removed

| Notebook | Reason |
| --- | --- |
| `analysis/easy_analysis.ipynb` | Earlier single-configuration regression analysis without the final readability/quality ensemble grid |
| `analysis/easy_analysis_readability-v0.ipynb` | Writes `union_output_v0.csv`; earlier combination enumeration |
| `analysis/easy_analysis_readability_v1.ipynb` | Earlier single-configuration, pre-generation workflow referring to v1 outputs |
| `analysis/final_analysis.ipynb` | Reads v0 summaries |
| `analysis/final_analysis_v1.ipynb` | Partial display of v2 ensemble summaries without the final single-predictor table and figure workflow |
| `analysis/final_analysis_v2.ipynb` | Earlier display over all context sizes; lacks the final Table 1/Table 2/Figure 3 presentation workflow |

## Other notebooks retained

These are distinct experiments or supporting tools, not established duplicate final versions. Their retention does not certify them as working release entry points.

| Group | Notebooks |
| --- | --- |
| Historical TREC DL/coherence and exploratory evaluation | `analyse_prototype`, `analyse_results_dl`, `analyse_results_dl-perplexity`, `coh_tests_trec-dl`, `qpp_test`, `qpp_test-perplexity`, `log_qpp_test`, `soft_rank_test`, `test` under `analysis/` |
| QPP preparation and experimental development | `analysis/nq_analysis_prototype.ipynb`, `analysis/from_test_to_experiment.ipynb` |
| Distinct downstream/example analysis | `analysis/downstram_analysis.ipynb`, `analysis/example_finding.ipynb` |
| Standalone significance/plotting exploration | `analysis/Fisher_test.ipynb`, `analysis/plotting/plot.ipynb` |
| Retrieval-score preparation | `analysis/retr_res/process_retrieval_score.ipynb` |
| Feature extraction and checks | Both readability notebooks; `qualt5_eval/quality_res/merge.ipynb`; `perplexity_eval/log_prob_temp_res/individual_with_query/check.ipynb`; `posterior_process/cal_0shot_posteriors.ipynb` |

## Conversion details

The experiment script preserves target construction, feature transformations, readability filtering, regression with intercept, and correlation calculations. Numerical behaviour is compared with the source notebook on deterministic synthetic NQ data: 36 single-predictor rows and 60 ensemble rows match to tolerance 1e-12 for one retriever/context size and both prediction targets. This test is not validation on original research data.

Operational changes: NQ defaults (historical DL remains selectable); configurable context sizes, retrievers, material directory and output directory; absolute paths independent of shell working directory; input preflight; new/empty output directory requirement; no notebook resume/appending to previous runs; `ast.literal_eval` replaces `eval` for serialized numeric lists; headless figures are closed; each model record has a unique feature-combination/pre-post key, ordered feature names, log1p feature names, coefficients and intercept.

The report script explicitly selects unique rows and rejects missing or duplicate results. It exports numeric CSV and plain LaTeX tables plus Figure 3 data and plots. It avoids the original undefined `rpp_df` and display-only notebook state. It includes the PerpA-only Table 2 row, which the original presentation notebook inspected separately. Plot styling and LaTeX decoration are not intended as pixel-identical publication reproductions.

Significance is exported separately as `notebook_significance.csv`. It preserves the helper formula and PostGen comparison against both PreGen and PerpA. It selects the query count from the matching configuration instead of an unfiltered first row. Significance marks and boldface in the publication have not been independently validated; numeric table checks do not certify them.

## Verification

```bash
python -m unittest discover -s tests
python tests/check_notebook_parity.py
```

The first checks all 84 published numeric table cells and rejects ambiguous input summaries. The second compares the actual converted experiment against the source notebook from Git history on controlled inputs (progress display alone is stubbed if tqdm is absent). Full model/data reconstruction remains blocked by missing external research inputs.
