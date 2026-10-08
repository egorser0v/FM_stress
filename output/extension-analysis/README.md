# Extension analysis: complete

96 / 96 declared fits. Original experiments remain unchanged.
Scores use global training-standardized channel units. Sources share channel correlation;
white means temporally white. Joint energy is divided by sqrt(F*C). Baselines use the
same evaluation origins. Error bars are optimization-seed SD, not confidence intervals.

`summary.json` contains all scalar metrics, per-PC velocity means, baseline values,
paired contrasts, geometry, capacity and OT-only comparison. Real context intervals
use circular moving blocks of 4 and 8 selected chronological origins; their agreement
does not prove independent observations or validity under arbitrary nonstationarity.
Many exploratory comparisons are unadjusted for multiplicity. Three seeds do not
establish population-level significance or model equivalence. Source comparisons
use proper forecast scores, never raw velocity errors with different labels.
Source-by-head interactions are (head GP - MLP GP) - (head white - MLP white),
paired before resampling. Negative values favor the head relative to MLP more
under GP; an interaction alone does not establish superiority in either cell.

`paths_<dataset>` shows the first three fixed evaluation contexts, first declared
seed, and last channel for all eight source/head cells with shared row axes.
Plots show marginal 90% intervals, not simultaneous path bands. ETTh1 OT-only
comparisons align forecast origins, target/scaler and channel; they do not compare
the seven-channel average with a univariate target. The joint-model conditioning,
output dimensionality, capacity allocation and source covariance still differ.

PNG and vector PDF figures are provided. This analyzer verifies provenance and
basic aggregation; final delivery additionally requires independent raw-array and
checkpoint recomputation. Partial output is never a final comparison.
