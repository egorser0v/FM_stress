# Exploratory calibration by observed-history variability

The analysis is motivated by aggregate center-only versus history-normalization coverage differences. Choices were fixed before inspecting any stratified forecast outcome. It is an exploratory diagnostic on the existing ETTh1 test period, not an independent confirmation or a new experiment.

The stratifier is the population standard deviation (`ddof=0`) of each observed lookback in the original train-standardized units, before per-window normalization. Cutpoints are the 25th and 75th percentiles of **training-history** standard deviations, using NumPy's linear quantile convention. Test histories are assigned to low (`std < q25`), middle (`q25 <= std <= q75`) or high (`std > q75`). Test information never sets the cutpoints. Exact training quartiles and configuration, data and code hashes are frozen in `results/followup-calibration-strata/manifest.json`.

The inclusion set is all 84 normalization/source/head/seed fits at both frozen budget selections, totaling 168 records. Report each bin's context count, fair CRPS, 90% interval coverage, interval width and realized ensemble-mean forecasting MSE. Retain every training seed and their equal-weight means. A single predeclared paired contrast is history+raw-unit-loss minus center-only, within source, head, seed, budget and bin. All five ridge pipelines use identical bins; their fixed residual-bootstrap forecasts do not have neural optimization-seed replications.

The data audit reconstructs training scaling, train/test windows and start indices directly from the CSV. It requires every saved forecast archive's target, chronological index and reconstructed original-unit history to match. Bin-weighted per-seed metric means must recover overall means. Final analysis refuses incomplete aggregates and requires the independent aggregate audit to match the consumed complete summary's hash. No stratified outcomes are exported before all 84 fits are complete.

Results go to `results/followup-calibration-strata/summary.json` and `per_seed.json`. All cell/bin means are exported to `output/followup-calibration-strata/strata.csv`; coverage, width and CRPS figures use the fixed final budget. No model is refitted and no source, head or threshold is selected using these results. History variability may describe where miscalibration occurs; it does not by itself establish a causal mechanism. Contexts belong to one chronological test period and three training seeds are not independent datasets.

```sh
.venv/bin/python scripts/analyze_followup_calibration_strata.py --prepare
# After all fits and the independent aggregate audit are complete:
MPLCONFIGDIR=/tmp/fm-followup-mpl .venv/bin/python scripts/analyze_followup_calibration_strata.py
```
