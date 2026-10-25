# Independent audit of final trajectory scoring

25 October 2026. **Passed; no blocking numerical or cohort-label error found.** Reviewed `fm_stress/trajectory_metrics.py`, `scripts/evaluate_trajectory_metrics.py`, `docs/trajectory_metrics.md`, the transfer manifest and the complete `results/trajectory-metrics/` outputs. This is a post-hoc scoring audit, with no training, model execution or new forecasts.

## Independent computation

Using NumPy 2.3.5 in the bundled Python runtime, a separate audit implementation read the original 15 ensemble archives and all 24 transferred forecast archives. It did not import the scoring module under review. It independently computed:

- Fair variogram score through the off-diagonal product identity `a²−2a sum(z)/M+((sum(z))²−sum(z²))/(M(M−1))`, rather than the production code's sample-variance subtraction.
- Plug-in variogram score; its finite-ensemble correction; direct full ordered-pair CRPS and energy scores, rather than the production sorted CRPS and upper-triangle energy implementations.
- First/second absolute finite differences for all ensemble paths and all transferred 1,024-path cohorts; every saved per-context vector, run mean, group mean and ratio.
- All paired-run differences and every saved 10,000-resample percentile interval, including recomputation of roughness ratios inside each resample. This includes main intervals and all within-draw follow-up intervals.

**1,877 numeric comparisons passed**, largest absolute discrepancy **3.552713678800501e-15**. All **90 consumed-file and consumed-ZIP-entry hash checks passed**; scored-code and output hashes also matched `results/trajectory-metrics/provenance.json`. This audit reconstructs score arithmetic and reported uncertainty from saved predictions. It does not independently reconstruct model training or establish data-ordering guarantees beyond the checked arrays and manifest.

## Scientific and cohort checks

The 39 models are exactly 15 original GP checkpoints, 12 GP-source label-noise checkpoints, and 12 white-source RBF/F64 capacity checkpoints. The label subset is not all 24 label-study fits; the capacity subset is not all 36 capacity fits. There are 11 groups and nine contrasts. Follow-ups use 32-member ensembles at Euler64; main uses 64-member ensembles at Euler128. Their absolute scores should not be compared as if they were one benchmark.

Shared transfer ensemble targets exactly equal the first 256 path targets. Single-path roughness uses a separate 1,024-context score namespace. Its R1/R2 values can differ from the ensemble-cohort values. In particular, the new capacity ensemble R1 ratio 2.7176 for the original MLP must not be silently substituted for the older single-path value around 2.677.

Weights are fixed, nonnegative, inverse time-lag weights normalized over unordered pairs. The fair variogram estimator is unbiased for the population variogram score under conditionally iid ensemble members and requires no covariance correction between time pairs. Negative estimates are retained. The variogram score is proper but not strictly proper; a common level shift is invisible. R1/R2 are moment diagnostics, not proper scores. These definitions and qualifications are correctly documented.

Main intervals resample 256 paired histories after averaging the five fixed models; they are conditional on these fitted models and saved ensemble draws. Follow-up intervals are explicitly conditional within each of three paired data/initialization draws. No pooled 768-history or three-draw population confidence interval is presented. Data and training-seed effects cannot be separated. Group ratio differences and means of per-draw ratio differences have different denominators and need not agree; the documentation acknowledges this.

## Claims the outputs support

- Original GP S4 has R1 ratio **1.0324** but R2 ratio **1.6276**, versus MLP **1.0066/0.9884**. Thus near-target first-difference roughness can hide excess second-difference roughness in these generated paths. It does not prove that every S4 forecast or conditional distribution is wrong.
- Main fair VS is **0.071953** (MLP), **0.069887** (S4), **0.068657** (whitened MLP). S4−MLP is **−0.002066**, interval **[−0.0043714, +0.00001999]**. The tiny positive upper endpoint must not be rounded into a claim of an interval excluding zero.
- Whitened MLP−MLP fair VS is **−0.003296**, interval **[−0.005310, −0.001420]**. This is an exploratory conditional-history finding with no multiplicity adjustment. Original CRPS/energy comparisons remain unresolved; this is not proof of superior overall calibration or a general whitening recommendation.
- In the transferred GP label subset, oracle supervision lowers mean VS for both heads; the MLP improvement has the same sign in all three draws, while the S4 contrast changes sign in one draw. S4 R2 remains above MLP under both supervision modes.
- In the transferred white/RBF/F64 capacity subset, all alternatives lower VS relative to the original narrow MLP in all three paired draws. R2 falls from **19.0884** to **1.6867** (rank-clear MLP), **1.6020** (wide MLP), and **1.0017** (S4). These diagnostics support a bottleneck-related shape problem, but near-one R2 alone cannot select the best predictive distribution. Alternative VS values are very close and the planned contrasts compare each to the original MLP, not all alternatives against one another.

No new success threshold or test-selected model recommendation is warranted. These scores were added after prior results had been inspected. The retained original success flags, narrow subset labels, finite Euler resolutions, and conditional uncertainty language must accompany presentation of the new figures.
