# Prospective conditional-ensemble evaluation

This supplemental protocol was fixed before evaluating the new draw. It does not
alter the primary experiment, success criterion, training or saved checkpoints.
The question is whether existing heads differ in conditional forecast quality,
which single-path mean roughness cannot establish. No winner is required.

Use `configs/ensemble_evaluation.json`: the main L=32, F=16, lengthscale-8 GP;
256 fresh independent joint history/future windows from seed 87041; 64 independent
GP-source samples per history from seed 88041; fixed Euler-128; batches of 256.
All three GP heads (MLP, S4 and whitened MLP) and all five original training seeds
use exactly the same histories, source samples and one observed held-out future
per history. Histories are repeated contiguously for the 64 ensemble members.
Only original validation-selected checkpoints are used; there is no new fitting,
checkpoint selection or selection based on this test draw.

## Metrics and estimands

All scores use normalized observation coordinates. Every history has equal
weight. Horizon coordinates are averaged within a history, never counted as
independent observations.

For an ensemble of size S, marginal **fair ensemble CRPS** is
`mean_s |X_s-y| - sum_{s<r}|X_s-X_r|/[S(S-1)]`, averaged over F coordinates.
This off-diagonal U-statistic removes the usual finite-ensemble bias in estimating
the population score. The sorted-ensemble identity computes the pairwise sum.
**Fair energy score** uses the same formula with Euclidean distances between
entire F-dimensional paths. Its scale is different from marginal CRPS; compare
heads within each metric. Lower proper scores are better. Individual estimates
are subject to ensemble Monte Carlo noise and one realized future per context.

Also report predictive-mean MSE averaged over F, central 80% and 90% interval
coverage and width, sample variance with ddof=1 averaged over F, generated and
target mean adjacent-jump roughness. Interval endpoints are empirical marginal
quantiles with NumPy's `method="linear"` (Hyndman–Fan type 7), at .1/.9 and
.05/.95. Coverage includes equality with either endpoint. Coverage is pooled
across histories and coordinates after calculating a per-history fraction; it is
not simultaneous path coverage. Width and variance require interpretation with
coverage and proper scores: smaller is not automatically better. Predictive-mean
MSE contains finite-ensemble mean-estimation noise and is not bias-corrected.

Report each training seed separately, head means and optimization-seed SDs. For
predeclared contrasts S4−MLP and whitened MLP−MLP, first average each per-history
score across the five checkpoints in a head, then bootstrap the 256 paired
history differences with 10,000 resamples (seed 90041), reporting percentile 95%
intervals. These intervals concern fresh-history variability conditional on the
fixed trained models and ensemble draws; they do not integrate training-dataset,
hyperparameter-search or GP-family uncertainty. They are exploratory, with no
multiplicity adjustment. Separate seeds' scores are not independent test sets.
The primary new comparisons are CRPS and energy score; all other scores diagnose
why those comparisons differ. This is not retrospective equivalence testing.

An optional reference draws 64 exact futures conditional on the full **raw**
history (seed 89041). It has information unavailable to the networks. Label it
“privileged exact-conditional GP reference”; it is neither an attainable model
baseline nor a finite-sample lower bound for estimated CRPS or energy score.
Its ensembles are exact conditional draws, not Euler trajectories.

## Fixed solver sensitivity

For all three heads at training seed 0, reuse the **first 32** of these histories
and their unchanged 64 source draws and generate Euler-256 ensembles. Compare
the same per-history metrics against the matching Euler-128 subset, plus paired
endpoint RMSE. This predetermined subset and step count are diagnostic; they do
not select the primary integration budget or replace the all-seed evaluation.

## Reproducibility

The evaluator refuses MPS requests when MPS is unavailable, saves checkpoint and
source hashes, the unchanged protocol configuration, common input arrays,
generated ensembles, per-history scores, per-seed records, and paired bootstrap
summaries. Completed files are reusable only when their fingerprint agrees.
Partial runs are explicitly incomplete; the final comparison requires all 15
checkpoints. No original experiment results are overwritten.

Run after CPU metric tests pass:

```sh
.venv/bin/python -m fm_stress.ensemble_evaluation --device mps
```
