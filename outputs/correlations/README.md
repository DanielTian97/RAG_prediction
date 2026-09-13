# Correlation outputs

This directory is reserved for the final numerical outputs produced by `analyse.py`.

- `single_feature_correlations.csv`: correlations of individual predictor signals with GPP/RPP targets.
- `feature_group_correlations.csv`: correlations of development-trained feature-family combinations with GPP/RPP targets.

The intermediate feature matrices produced by `compute_features.py` live under `outputs/features/` and are intentionally gitignored.

The two `paper_*.csv` files retained in this directory are the final ECIR-era result summaries from the original repository, kept as reference outputs while the refactored pipeline is validated against them.
