# Independent audit against the supplied project assignment

**Status: all computational deliverables completed.** The independent subagent review verified the 25-run primary experiment and scientific design. The primary agent subsequently verified the 15-run shorter-lengthscale series, nine warm restarts and saved-checkpoint solver checks after the subagent service reached its usage limit. Final verification recomputes metrics from archived arrays, checks validation selection and source hashes, and validates the report. This provenance distinguishes independent review from the final implementation audit; it does not claim that the unavailable subagents reviewed the last PDF.

Reviewed against both supplied PDFs: the four-page assignment (Tasks 1/2/3, weights 30%/40%/30%) and the three-page project plan. Reviewed implementation: `data.py`, `metrics.py`, `models.py`, `experiment.py`, `analyze.py`, `robustness.py`, configurations, scientific protocol, provenance and tests.

## Requirement coverage

| Requirement | Completed evidence |
|---|---|
| Task 1 critical synthesis, Sundial/TSFlow/S4 and missing cell | `related_work.md` and the one-page `related-work.pdf`; published systems are distinguished from tiny adaptations |
| Joint GP windows, independent sources, fixed F and Sundial normalization | L=32, F=16; history-only population mean/std and exact small-std fallback; independent RNG streams |
| Train-fitted PCA, held-out spectra and pre-training gate | Both saved geometry records pass; tables and spectra exported for both data configurations |
| Task 2 four source × head combinations | 20 primary grid runs plus 12 follow-up grid runs; matched initialization, draws and update budgets |
| Three metrics and generated/target paths | Complete CSV/JSON tables and predetermined-context figures; original-coordinate scoring; PC2 retained separately |
| Task 3 whitening control | Five primary and three follow-up control runs; train-only PCA, consistent derivative transform and inverse map |
| Mandatory recommendation | `docs/conclusions.md`; report page 11; formal method verdict withheld after the all-fail branch |
| Reproducibility and MPS | All 40 base runs and nine warm restarts on MPS; checkpoints, arrays, logs, hashes and dependencies saved |
| Numerical and visual verification | 26 passing tests; array-to-metric recomputation; 11-page report rendered and inspected |
| Optional real series and oral work | Real series explicitly omitted; oral participation remains the team's responsibility |

## Scientific findings that must remain visible

**The literal success flag is not a general accuracy test.** The assignment says residual error should be “within 20%” of PC1 error without specifying scaling or one-sided versus two-sided comparison. The protocol fixed the raw, per-component, two-sided version before the scored runs. It can reject exceptionally small residual error because raw PC variances differ greatly. Keep its results unchanged; show variance-normalized values as secondary diagnostics, rather than silently redefining a success.

The completed analytic reference supports this concern independently of learned rankings. `results/main/oracle.json` gives raw residual/PC1 ratios **0.0154286** (white source) and **0.0039870** (GP source). The exact conditional GP sample roughness ratio is **0.9798264**. Thus the reference fails the raw equality flag while its path roughness is target-like. Its GP normalized error ratio is **1.1721536**, but this does not authorize replacing the primary rule after the fact.

That reference is **privileged and composite**: its velocity conditions on the full raw history, while networks receive normalized history; its roughness comes from exact GP draws, not an Euler solution of the analytic field. A separate solver diagnostic now exists. At Euler-64, `results/diagnostics/oracle_solver.json` reports analytic expected roughness ratios of approximately **0.9893** (white) and **0.9916** (GP) relative to the exact conditional distribution. Nevertheless, about **82.4%** of white-source conditional covariance directions and **12.2%** of GP-source directions retain less than 80% of target variance. Good mean roughness therefore does not establish distributional accuracy or negligible solver error. These are oracle diagnostics, not learned-model results.

**The S4 comparison is genuine but deliberately reduced.** The implementation uses HiPPO-LegS, a learned rank-one NPLR correction, bilinear discretization and three bidirectional residual blocks. Independent complex-reference tests check outputs and gradients, and the temporal convolution is checked against the official FFT convention. It is not S4D or a generic CNN proxy. The state dimension is 16 rather than the inspected TSFlow default 128, and conditioning comes through the shared encoder design. Both adaptations are documented. The MLP and S4 parameter counts are 65,464 and 65,281. Equal updates and near-equal parameter counts do not establish equal optimal tuning or identical computational cost; report actual budgets and wall times.

**“Same encoder” means design and initialization.** Runner-level tests verify identical initial encoder weights and paired first training draws. The encoders then train separately. An eventual difference includes interaction between head and encoder; it does not isolate only the final head Jacobian. There is no pretrained backbone.

**Whitening is not dimension removal.** The control retains all 16 directions, with a declared eigenvalue floor. It rescales inputs, velocity targets and the optimization objective. Metrics are computed after mapping back. A favorable control result could support this intervention, but not prove that unused coordinates or a particular Jacobian mechanism caused the difference.

**The source matches the raw kernel, not the normalized marginal covariance.** Context normalization creates a mixture of conditional Gaussians. The source remains independently drawn in normalized observation coordinates. The protocol accurately discloses this distinction; final captions and conclusions should retain it when interpreting source effects.

## Verified primary findings (five paired optimization seeds)

These statements refer only to the recorded main configuration, fixed synthetic data draw and 6,000-update budget. The numbers below were independently recomputed from `results/main/runs/*.json` and agree with the analysis files. No cross-source velocity-MSE ranking is inferred because source changes alter the regression-label distribution and PCA basis.

| Cell | PC1 MSE, mean | Residual MSE, mean | Roughness / target, mean | Raw rule passes |
|---|---:|---:|---:|---:|
| White noise + MLP | 6.677114 | 0.122113 | 1.057679 | 0/5 |
| White noise + S4 | 7.320931 | 0.124918 | 1.044564 | 0/5 |
| GP + MLP | 14.929345 | 0.049272 | 1.018412 | 0/5 |
| GP + S4 | 15.502381 | 0.058269 | 1.038528 | 0/5 |
| GP + whitened MLP | 13.618078 | 0.040097 | 1.016022 | 0/5 |

All 25 runs pass the roughness band, but none passes the raw error-equality condition. This is not evidence that all generated paths are poor. The oracle diagnostic above already shows that the equality requirement can reject an appropriate reference. The assignment nevertheless requires retaining the all-fail result and performing the separate geometry-audit branch.

For the GP source, the central S4-advantage prediction is **not supported by this experiment**. S4's mean PC1 error exceeds MLP's by 0.5730 (paired percentile 95% bootstrap interval [0.0491, 1.0969]); its residual error exceeds MLP's by 0.008997 ([0.008219, 0.009775]). The latter difference has the same direction in every seed. This does not establish an inherent disadvantage of S4: its reduced state size, common optimizer budget and encoder interaction remain relevant limitations.

Whitening improves the GP MLP's two velocity errors in every paired seed. Relative to the raw GP MLP, mean PC1 MSE is 8.78% lower and residual MSE is 18.62% lower. The paired differences are −1.3113 ([−1.5552, −1.0674]) for PC1 and −0.009175 ([−0.009609, −0.008550]) for residual MSE. Its roughness remains inside the band in all five runs. A distinct roughness advantage is not established: the interval for the difference in absolute relative roughness deviation spans zero (whitened minus raw MLP: [−0.03376, 0.01369]). Use “improves velocity errors while preserving acceptable roughness,” not “wins every metric.”

The white-source comparison similarly provides no S4 velocity advantage: its PC1 difference is +0.6438 ([0.4799, 0.8111]); the residual difference interval spans zero. These within-source findings do not prove equivalence between heads, and the bootstrap intervals are descriptive of five optimization seeds, not broad statistical guarantees or multiplicity-adjusted tests.

The normalized diagnostic passes in 5/5 white-MLP, 4/5 white-S4, 0/5 GP-MLP, 0/5 GP-S4 and 4/5 GP-whitened runs. These are secondary flags only. Ten of the GP velocity eigenvalues are below the declared variance floor; disclose this when interpreting normalized residual values. In particular, small absolute errors in those directions can become large relative errors.

As a sanity check, leaving the initial source unchanged gives roughness ratios 5.9192 (white) and 0.5250 (GP), using the saved seed-0 source arrays and targets. Thus a smooth GP source alone does not automatically meet the target roughness criterion in this setup. This no-transport baseline is not a conditional forecaster and does not establish distributional calibration.

**Primary-only recommendation:** keep the small MLP as the baseline and carry the whitened MLP forward as the promising coordinate intervention; these data do not justify switching to the tested S4 head. This is a provisional experimental recommendation, not a declaration that any head passes the literal primary rule or that whitening is universally superior. The final wording in `docs/conclusions.md` incorporates the completed repair and optimization/solver diagnostics and does not claim that a Jacobian mechanism was demonstrated.

## Final completion checks

1. All 25 primary and 15 shorter-lengthscale runs completed on MPS, with checkpoints, samples, training curves and consistent source/configuration hashes. The saved-data verification is in `results/delivery_verification.json`.
2. The shorter-lengthscale follow-up reran Task 1 before training: GP-velocity test E2=0.94605 and white-velocity E2=0.79000. All 15 runs pass roughness, and none passes the raw equality condition. It therefore does not rescue a formal method conclusion.
3. All nine lower-rate warm restarts completed on MPS. The validation-only expansion trigger is preserved. Mean test raw velocity MSE is 1.12851 (MLP), 1.17660 (S4), and 1.04505 (whitened MLP). This supports the direction of the primary coordinate-control result while showing sensitivity to finite optimization budgets.
4. Saved-checkpoint Euler checks use CPU, the first 256 test histories and seed 0; training remains MPS. Changing K=64 to K=256 changes endpoints by 1.31–1.79% of target RMS. This does not negate the separate analytic covariance/calibration concern. These subset roughness values must not be confused with the primary 1,024-path estimates.
5. The mandatory related-work note, complete primary and follow-up tables, fixed-context plots, one-paragraph recommendation, 11-page report, source provenance and editable code are provided. The optional real-data extension is explicitly omitted. The original proposal poster is preserved separately.
6. All 26 numerical tests passed again with actual MPS access (`results/final-test-log.txt`). The final report was rendered page by page; the primary metric table and solver page received detailed visual inspection. Reproducibility scripts reject reuse of changed numerical sources.

No blocking mathematical implementation error was found. The computational work for Tasks 1–3 is complete; the formal method verdict remains inconclusive under the assignment's retained criterion. The shorter-lengthscale S4 has slightly lower GP PC1 error than MLP but higher residual error, so no across-metric or universal head ranking is justified. `docs/conclusions.md` is the final recommendation. Team members must still understand the work and perform the oral presentation themselves.
