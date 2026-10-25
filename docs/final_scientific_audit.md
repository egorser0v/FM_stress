# Scientific audit for the final paper and defense

Audit date: 25 October 2026. This is an independent interpretation and arithmetic review of the archived main and follow-up evidence, not a new training experiment. Historical reports and declared success flags remain intact. The 144 follow-up fits comprise 36 capacity, 84 normalization and 24 label-supervision fits; checkpoints at two budgets do not double the fit count.

## Defensible central claim

**The proposed universal GP-source disadvantage of an MLP head is not established. Apparent head advantages depend on the diagnostic, decoder capacity, supervision and normalization pipeline.** The evidence supports specific limitations of comparisons and metrics, rather than a replacement architecture or a universally best prior.

This is stronger and more useful than either “nothing worked” or “S4 wins”: the original equality criterion is not discriminating, the F64 decoder restriction is exact, and controlled interventions reveal which findings persist. The original F16 result and supplementary F64/real-data studies must remain distinct.

## Numerical anchors and their meaning

All values below come from the indicated JSON, with arithmetic over stored per-PC records reproduced by `python3 scripts/audit_final_pca.py`. That script hashes inputs and writes `results/final-pca-audit/summary.json`; it does not independently regenerate predictions.

### Original F16 GP-source experiment

| Head | PC1 velocity MSE | PC3–16 mean MSE | Mean PC3–6 MSE | Fair CRPS on fresh histories |
|---|---:|---:|---:|---:|
| MLP | 14.9293 | 0.049272 | 0.172429 | 0.466306 |
| S4 | 15.5024 | 0.058269 | 0.196926 | 0.461315 |
| Whitened MLP | 13.6181 | 0.040097 | 0.140330 | 0.460568 |

Velocity values: `results/main/runs/gp_*_seed*.json`, independently collated in `results/diagnostics/report_addendum.json` and the final PCA audit. CRPS: `results/ensemble-evaluation/summary.json`, five fixed trained seeds, 256 fresh histories and 64 independent source draws per history. These are two evaluation sets and must not be portrayed as one matched table of per-run outcomes. All four originally declared CRPS/energy contrasts against MLP include zero in their paired-history intervals. Failure to resolve a difference is not equivalence.

The original 25/25 roughness passes and 0/25 literal successes stand. The two-sided rule requires residual error to be close to PC1 error, so a much smaller residual is counted as failure. A privileged oracle also fails. Thus the rule measures the assignment's chosen relation between errors; it is not a test of calibrated forecasting or architectural competence. Preserve it as the historical endpoint and explicitly mark later scores as supplementary.

### Signal and tail require different descriptions

The archived addendum overstates its sentence that all PCs7–16 “carry no structure.” Its own target PC7 eigenvalue is 2.1848e-5, about 3.91 times the estimated normalized nugget 5.5822e-6. PCs8–16 range from 5.2636e-6 to 6.0006e-6 and are nugget-dominated. PC7 is a transition direction. Ten floored directions are not proof of ten signal-free directions.

Report PC1, PC2, PC3–6, PC7 and PC8–16, while retaining the original PC3–16 metric. Source-specific velocity PCs and target PCs are different bases. A common PC index across different sources does not identify a common physical subspace; use fixed target PCA for cross-source illustrations.

For GP+MLP, PC3 contributes 86.6% of the original raw residual sum. The normalized S4 residual metric is 91.8% attributable to PCs7–16, and 92.7% in the raw-loss controlled experiment. Consequently the dramatic normalized improvement under balancing mostly concerns a floor-sensitive, very low-energy tail. It is real behavior but does not by itself establish better learning of meaningful curvature or better forecasts. Source: `results/diagnostics/report_addendum.json`.

### Information-matched oracle and excess risk

On the same fresh 2,048 histories, using a reference conditioned on full normalized history:

| Head | PC1 model–oracle squared discrepancy | PC3–6 discrepancy | PC3–16 discrepancy | Equal-direction residual Jacobian error | Covariance-weighted error |
|---|---:|---:|---:|---:|---:|
| MLP | 6.26096 | 0.057731 | 0.016499 | 0.98935 | 0.60364 |
| S4 | 6.22923 | 0.076748 | 0.023932 | 0.95113 | 0.64639 |
| Whitened MLP | 5.40996 | 0.022755 | 0.006502 | 3.03286 | 0.36522 |

Velocity and derivative evaluations use different context sets; column labels must retain this distinction. Source: `results/followup-oracle/summary.json`, `summary` and `velocity_rows`; numerical checks in `results/followup-oracle/independent_verification.json`.

The matched oracle's empirical PC1 label MSE is 9.66445, not the earlier privileged-reference value near 7.9 suggested in the review. The reference, context set and normalization matter. Do not subtract the old oracle value from a new model MSE. Population excess risk equals squared model–conditional-mean discrepancy, but the finite test-set subtraction also contains a signed cross term. For example, MLP PC1 label MSE 15.36981 minus oracle label MSE 9.66445 is 5.70536, whereas direct discrepancy is 6.26096. These quantities must not be silently equated. The stored cross-term decomposition is retained in the PCA audit.

Correcting the information set does not remove the metric-dependent Jacobian ranking: S4 is closer in an equal-direction norm; MLP is closer after covariance weighting; whitened MLP has the largest unweighted error but the smallest weighted error. The weighted metric describes typical local perturbations, not a proven optimizer mechanism. The oracle sees complete normalized history; the network sees its learned encoding, so an encoder limitation remains possible.

### Strongest controlled extension: F64 capacity

The original width-56 decoder at horizon 64 has at least eight fixed directions: for a projection N orthogonal to the output matrix, Nᵀv=Nᵀb and therefore Nᵀx(1)=Nᵀx(0)+Nᵀb. This is a global architectural identity. It does not apply as a dimension obstruction to the original F16 experiment.

At 12,000-update prefix-selected checkpoints, fresh RBF/white residual MSE is 0.464370 (MLP56×3), 0.354633 (MLP65×2), 0.342281 (MLP72×3), and 0.326246 (S4). Corresponding CRPS is 0.635215, 0.630911, 0.631838 and 0.632137. The near-parameter-matched and wider controls close approximately 79.4% and 88.4% of the original residual gap, using the ratio of mean gaps across the three draws (the mean of per-draw fractions rounds to the same displayed percentages). This is not the percentage improvement in CRPS. A residual S4 advantage remains; forecast differences are small and can be solver-sensitive.

Sources: `results/followup-reproduction/decoder_span.json`, `output/followup-capacity/summary.json`, `results/followup-solver/summary.json`. Wider models change parameterization and, for width65×2, depth. These controls support a bottleneck contribution, not a claim that all differences are caused by rank alone. Three paired data/optimization draws do not separately identify data and initialization variances.

### Label-noise intervention

Conditional-mean supervision reduces residual discrepancy to the matched oracle in all 12 paired source/head/draw contrasts. Mean ratios oracle-label/stochastic-label are 0.79584 (white/MLP), 0.89153 (white/S4), 0.83201 (GP/MLP), 0.86079 (GP/S4). These are means of per-draw ratios, not ratios of pooled means. At the original Euler64 resolution CRPS improves in all six MLP pairs and four of six S4 pairs; full Euler128 sensitivity must accompany any tiny head-specific contrast. Eleven of 24 validation selections occur at the final step. Convergence is not established.

Source: `output/followup-label-noise/summary.json`; full solver records: `output/followup-label-solver/summary.json`. This intervention supplies privileged generator knowledge and uses oracle-based validation for both labels. It tests the effect of removing conditional label variance in this pipeline, not a deployable forecast method or proof that stochastic gradients are the unique limitation.

### Real-data baseline fairness

The older ETTh1 history-normalized comparison reports ridge CRPS 0.085185 and white-source MLP 0.085510 on its 64 contexts. It does not show a neural gain over ridge. The new 118-context follow-up gives history-normalized ridge 0.093624, compared with 0.098702/0.106788 for white MLP/S4 and 0.098470/0.100514 for GP MLP/S4. Keep ridge visible in the main real-data table. Do not compare the old 64-context 0.0852 with the new 118-context neural scores.

The new global/GP MLP–S4 contrast is 0.216480 versus 0.174803; center-only gives 0.098211 versus 0.098071. This is a pipeline-dependent interaction, not a normalization-independent S4 win. Similar CRPS hides substantial width/coverage differences, especially in quiet histories. Sources: `results/etth1-normalization/summary.json`, `output/followup-normalization/summary.json`, `results/followup-calibration-strata/summary.json`.

The history-normalized ETTh1 GP-velocity top-two validation share is 0.821–0.851, below the original 0.90 gate. Label this supplementary real-data evidence, not completion of a qualifying optional original task. The previously inspected test period is not untouched; overlapping or dependent windows are not independent datasets.

## Metrics audit for the final supplemental scoring

For ensemble size M, coordinate pair i<j, z_m=|x_mi−x_mj|^p and a=|y_i−y_j|^p, the unbiased finite-ensemble variogram contribution is

`(a − mean(z))² − var(z, ddof=1)/M`.

Equivalently it is `a² − 2a sum(z)/M + ((sum(z))² − sum(z²))/(M(M−1))`. Use fixed nonnegative pair weights and state whether they sum to one. The expectation equals the population variogram score for conditionally independent ensemble samples. Individual fair estimates may be negative; clamping introduces bias. A deterministic forecast needs its deterministic score, not division by M−1 with M=1.

The variogram score is proper but not strictly proper and is insensitive to a common level shift. It complements marginal CRPS and energy score; it cannot establish full calibration by itself. p=0.5 with fixed weights is a defensible declared secondary diagnostic. Do not choose p, weights or scored models after seeing favorable results. Definition: [Scheuerer and Hamill, 2015](https://repository.library.noaa.gov/view/noaa/22327/noaa_22327_DS1.pdf). Explicit fair-ensemble formula: [scoringrules documentation](https://scoringrules.readthedocs.io/en/latest/crps_estimators.html).

R1/R2 roughness ratios should be reported with their actual first/second finite-difference definitions and denominators. A ratio near one matches one unconditional moment. The GP no-flow control already has R1 ratios 0.846 (RBF) and 0.901 (OU), yet CRPS 0.734 and 0.673 and zero original-rule successes. Thus smoothness can come from the source alone. Do not create a new retrospective pass rule from R2 or VS. Source: `results/followup-capacity-baselines/summary.json`.

## Recommended 12-minute defense

| Time | Point | Evidence to show |
|---|---|---|
| 0:00–1:00 | Question and narrow hypothesis | White/GP × MLP/S4 diagram; explicitly adapted small heads |
| 1:00–2:00 | Why this is a conditional flow experiment | x_t=(1−t)ε+ty, target y−ε; observed normalized history |
| 2:00–3:30 | Original negative result | Four original cells and whitening; all formal failures; raw residual favors MLP |
| 3:30–5:00 | Why the original metrics cannot settle forecasting | PC spectrum, PC3–6 versus tail; roughness counterexample; fresh CRPS with uncertainty |
| 5:00–6:30 | A concrete confound with an exact explanation | F64 decoder identity and width controls; label F64 supplementary |
| 6:30–8:00 | Reference and supervision matter | Matched oracle metric reversal; 24-fit label intervention with residual discrepancy |
| 8:00–9:15 | Real-data transfer with an honest baseline | Global versus center-only GP table; ridge; calibration widths |
| 9:15–10:15 | What remains unresolved | Fixed budgets, conditioning asymmetry, only three synthetic draws, reused real test period |
| 10:15–12:00 | Contributions and final answer | Reproducible stress-test + metric audit + controlled explanations; no universal winner |

For ten minutes, move the oracle derivation and mixture sweep to backup. For fifteen minutes, add two minutes on covariance-mixture results and one on solver sensitivity. Do not show every training run as a separate achievement. Backup slides should contain the full original rule, oracle decomposition, all covariance-mixture levels, solver changes, parameter counts, ridge setup and contribution/AI disclosure.

## Concise conference-paper structure

Use a title such as “When Smooth Priors Do Not Require Temporal Heads: A Controlled Flow-Matching Stress Test.” Avoid a title claiming an intrinsic failure or a general forecasting improvement.

1. Introduction: the proposed prior–head incompatibility, the absence of a clean causal comparison, and three contributions: controlled stress test; audited diagnostics; targeted controls.
2. Setup and evaluation: independent coupling, observed information, kernel/normalization, head adaptations, original criterion, supplementary proper scores and paired replication units.
3. Main result: original F16 table and spectra; explain the all-fail criterion and unresolved proper-score comparison.
4. Explanatory controls: F64 decoder theorem and capacity controls as the strongest concrete mechanism; matched oracle and label intervention as a bounded explanation of remaining field error.
5. Transfer and limits: one compact normalization/ridge table; mixture and kernel scan in appendix. Real-data findings are supplementary and test reuse is disclosed.
6. Discussion: no architecture mandate; objective and normalization matter; requirements for future work. Preserve negative results and unresolved mechanisms.

Four main figures suffice: experiment/spectrum, main directional errors plus CRPS, decoder/capacity control, and normalization/ridge. Keep oracle derivation, all 144 follow-up cells, additional kernels, mixture levels, floor sensitivity rationale and solver details in the appendix. The paper and slides should be generated from the same numeric source registry, with evaluation context and aggregation unit attached to each value.

## Remaining limitations and disclosure

No F16 matched AdaLN-S4 conditioning ablation, broad per-head learning-rate search, complete convergence study, or prospective whitening floor/shrinkage sweep is present. Identity conditioning removes the learned-history-encoder difference in one follow-up, but not every head-conditioning difference. Equal nominal budgets are a bounded comparison, not proof of equal attainable performance. No whitening recommendation should rest on velocity error alone.

The user has approved the role statements in `docs/contributions.json`; commit authorship or agent logs were not used to establish intellectual ownership. State the actual known use of AI assistance in code, experiment execution, audit, visualization and drafting, and distinguish human decisions and verification. Do not invent individual roles, claim AI-produced audits were independent human peer review, or describe synthetic-label supervision as an ordinary deployable baseline. Independent software checks are evidence of arithmetic/provenance, not a substitute for external validation.

## Final coverage revision, 25 October 2026

The delivered defense now has 14 main slides and four backups; its 14:05 planned budget supersedes the earlier proposed outline above. Normalization is main slide 10, vector-valued multivariate prediction is main slide 11, and the explicit four-row reviewer response is main slide 12. Paper Section 6 restores both source families, three-channel synthetic GP, ETTh1 OT/seven channels and Exchange/eight channels, with CRPS, scaled joint energy and matched ridge-residual baselines.

The multivariate extension uses global TRAIN-channel normalization without per-window centering/scaling, 2,000 updates and 64 contexts ×32 forecast members. It must remain distinct from the 118-context/6,000-update normalization intervention. GP-source MLP/S4/whitened CRPS is 0.194988/0.143717/0.132086 for OT, 0.375027/0.403335/0.529772 for ETTh1 seven channels, and 0.385196/0.425454/0.564036 for Exchange eight channels. Whitening hurts both real multivariate tasks under both sources; the paper retains the opposite univariate/synthetic cases. Ridge with whole-vector residual resampling outperforms every displayed neural cell in CRPS and scaled joint energy. These are fixed-budget observations, not an isolated effect of channel count or a universal architecture ranking.

Canonical sources: `output/extension-analysis/summary.json`, `output/sparse-analysis/summary.json`; protocol: `docs/extension_protocol.md`. An independent table audit matched all 48 neural entries and eight unique ridge entries to displayed precision. Current all-18-slide visual and coverage review is in `docs/presentation_visual_audit.md`. The preposter supplies problem formulation only; no experimental results are attributed to it.

The response now explicitly separates external Euler resolution, flow-time embedding/conditioning, and S4 internal learned Δ. The latter initializes in [0.001, 0.1] but is not constrained to that interval during learning; learned-Δ inspection is not an initialization-range sweep. Matched conditioning, broad LR tuning, flow-time ablations and prospective floor/shrinkage/internal-Δ-range sweeps remain open.

The approved contribution statements now appear in the paper and on main slide 14. Its 30 seconds replace the former contribution reserve, preserving the 14:05 planned total. No scientific result changed in this role-statement revision.
