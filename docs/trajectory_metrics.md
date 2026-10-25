# Post-hoc trajectory scores on archived forecasts

This descriptive supplement computes new scores without training, checkpoint
selection, model loading, or generating a forecast. It introduces no success
threshold and does not change any original result. The canonical output is
`results/trajectory-metrics/summary.json`; accompanying CSV files provide run,
group and paired-contrast views. `per_context.npz` retains the exact score vectors
used in the analysis. `provenance.json` hashes every consumed input file/archive,
each consumed transferred ZIP entry, the scoring code and the outputs.

## Cohorts and alignment

* **Main:** 15 archived GP models, three heads × five optimization seeds, evaluated
  on the same 256 held-out histories with 64 ensemble members and horizon 16.
  Inputs and source draws are shared across all models. These five training seeds
  are not five independent test draws.
* **Label-noise follow-up:** 12 transferred GP models, MLP/S4 × stochastic/oracle
  training labels × three paired data-draw/training-seed replicates. Each archive
  contains 256 ensembles of 32 members, horizon 16, and 1024 separately drawn
  generated paths.
* **Capacity follow-up:** 12 transferred RBF, white-source models, original MLP,
  rank-clear MLP, wide MLP and S4 × three paired replicates; same forecast counts,
  horizon 64. Completed training budget is 12000 and checkpoints are validation
  selected. Label-noise completed budget is 6000. Generation used the original
  Euler budgets (main 128, follow-ups 64); no new solver comparison is made.

The transfer ZIP is read in place without extracting checkpoints or large test
files. All archive/entry hashes are checked against the transfer manifest. Each
forecast and shared target/history array is checked for shape, dtype and value
hash. `ensemble_target == path_target[:256]` is checked exactly. Within a
family/data seed, each model uses the same shared targets and histories. Context
indices have no correspondence across independent data draws. Main targets come
from the unchanged `common_inputs.npz`. Recomputed per-context fair CRPS, fair
energy, generated R1 and target R1 must reproduce the archived values within the
existing float32 numerical precision (`rtol=1e-6`, `atol=1e-10`). These are
reproduction/alignment checks, not newly proposed scientific pass criteria.

## Definitions

All calculations use float64 after loading the saved arrays. Units are the
original dimensionless history-normalized observation coordinates; per-window
raw location/scale is not available in the transfer. Compare models within a
suite, not absolute score magnitudes across horizons or data families. Every
history has equal weight. Members and time coordinates are not independent
replicates for uncertainty estimation.

For a path `x[0:F]` with unit time spacing:

* `R1(x) = mean_t |x[t+1]-x[t]|`.
* `R2(x) = mean_t |x[t+2]-2*x[t+1]+x[t]|`.

Ensemble R1/R2 first average the path value over members, then over histories.
The reported ratio is `mean_generated_Rk / mean_target_Rk`, not the mean of
per-path ratios. Group ratios are recomputed from group generated and target
means. The 1024 single-path cohort is reported separately for follow-ups; it is
not silently mixed with the 256-context ensemble cohort. A ratio near one only
matches a roughness moment and cannot establish forecast quality. Roughness is
descriptive and is not a proper forecast score.

The variogram score follows the variogram-moment construction of
[Scheuerer and Hamill (2015)](https://repository.library.noaa.gov/view/noaa/22327/).
Use order `p=0.5`, all **unordered** time pairs `i<j`, and
`w_ij = (1/(j-i)) / sum_(i<j)(1/(j-i))`, so the weights sum to one for each
horizon. The normalized unordered-pair convention must be retained when
comparing these numbers with another implementation.

For `z_mij=|X_mi-X_mj|^p`, `a_ij=|y_i-y_j|^p`, ensemble size `M`:

```
VS_plugin = sum_(i<j) w_ij * (a_ij - mean_m z_mij)^2
correction = sum_(i<j) w_ij * var_m(z_mij, ddof=1) / M
VS_fair = VS_plugin - correction
```

The correction is a direct finite-ensemble bias calculation: conditional on the
forecast distribution and observed target, `E[(a-zbar)^2] = (a-E[z])^2 +
Var(z)/M` for iid members. Sample variance with `ddof=1` estimates `Var(z)`.
No variance of different time pairs is needed, since the score sums squared
individual moment errors. The reported fair estimate is unbiased for the
population variogram score under conditionally iid draws. It can be negative
and is never clipped. The correction removes ensemble Monte Carlo bias; it
does not address model error or uncertainty from a single observed future.
Variogram scores are proper but not strictly proper: common path shifts are
invisible. Read them together with marginal CRPS and whole-path energy.

For absolute distance `d` (coordinatewise and then averaged over time for CRPS;
Euclidean distance over the entire path for energy):

```
plugin = mean_m d(X_m,y) - sum_(m<n) d(X_m,X_n) / M^2
fair   = mean_m d(X_m,y) - sum_(m<n) d(X_m,X_n) / (M*(M-1))
```

These retain the existing repository's fair CRPS and **unscaled** fair energy
definitions. Energy is not divided by `sqrt(F)`. Lower proper scores are better;
negative finite-ensemble fair values are valid. Plug-in/fair pairs are both
retained to show the effect of ensemble size.

## Aggregation and uncertainty

All runs, group means and run SD/min/max are saved. Contrasts are exploratory,
`a minus b`, with no multiple-comparison adjustment. Main contrasts are S4−MLP
and whitened MLP−MLP. Follow-up label-noise contrasts compare oracle−stochastic
within each head and S4−MLP within each label mode. Capacity contrasts compare
each alternative against the original MLP. No model is selected using these
post-hoc scores.

For main contrasts, report the five paired optimization-seed differences.
Average the per-context scores over all five fixed models within each head,
then bootstrap the 256 paired contexts (10000 percentile resamples, seed 90041).
Intervals condition on the five trained models and saved ensemble draws;
they do not include training-dataset or hyperparameter uncertainty. A roughness
ratio is recomputed from resampled generated/target means on every draw.

For each follow-up contrast, report all three independent paired replicate
differences and their descriptive mean/SD/min/max. A replicate combines a data
draw and a training seed, so those effects cannot be separated. Also save an
explicitly conditional paired-context bootstrap interval within each replicate.
There is no pooled context CI across replicates, no claim that 768 contexts are
independent of fitted models, and no population CI from only three replicates.
Group ratio differences and the mean of replicate ratio differences can differ
because the target roughness denominator varies between data draws; both are
derivable from the saved canonical group and paired-run values.

## Reproduce

Only Python and NumPy are needed; no torch, scipy, accelerator, checkpoint load,
network call or full archive extraction occurs in the evaluator.

```sh
python scripts/evaluate_trajectory_metrics.py
python -m pytest tests/test_trajectory_metrics.py
```

Original archives and result files are read only. All new artifacts live under
`results/trajectory-metrics/`. Hashes identify the exact inputs; they cannot
establish context ordering beyond the archive's documented alignment and the
explicit target-prefix/value checks.
