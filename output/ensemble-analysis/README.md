# Conditional ensemble evaluation figures

All 15 original GP checkpoints are included. Each shares 256 new histories and 64
source draws per history; Euler-128 is fixed. Scores and intervals are supplemental.
The main experiment and its success rule are unchanged.

- `per_seed_metrics.csv`: every checkpoint's mean metrics and runtime.
- `summary_metrics.csv`: metric means and SD across the five fixed trained seeds.
- `paired_context_contrasts.csv`: S4−MLP and whitened MLP−MLP, paired-history
  bootstrap after averaging context scores across training seeds. Intervals are
  conditional on these trained models and ensemble draws, with no multiplicity
  correction. Histories—not coordinates or ensemble members—are sampling units.
- `proper_scores`: CRPS and energy score; individual seed values and head means.
- `coverage_width_table`: marginal interval calibration and spread.
- `fixed_context_forecasts`: first three fresh contexts, original seed 0, with
  consistent y axes across heads within each row. Bands are marginal intervals;
  the single observed future need not equal the predictive median.
- `privileged_reference_metrics.csv` and `privileged_reference_paths`: exact
  conditional GP given RAW history, information unavailable to the networks.
  This reference is not an attainable baseline or finite-sample lower bound.

Figures are exported as PNG and editable vector SVG. `analysis_status.json`
records the exact input fingerprint. Raw ensembles remain in the results folder.
