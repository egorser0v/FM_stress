# Assignment traceability and scientific audit

Authority: the supplied four-page project assignment, *Smooth source, unstructured head: a flow-matching stress test*, defines Tasks 1–3 and weights 30% / 40% / 30%. The supplied three-page project plan adds independent splits, team responsibilities and reproducibility checks. The poster rubric is a separate presentation assessment. The matrix specifies acceptance evidence; the completion record below links each task to the finished artifacts and saved results.

| Task | Requirement | Acceptance evidence |
|---|---|---|
| 1 — 30% | Critical synthesis of TimeFlow/Sundial, TSFlow and S4; identify missing smooth-source/MLP comparison | `docs/related_work.md`; explicit distinction between head comparison and pretrained-model reproduction |
| 1 | Fix a small horizon before experiments; sample lookback and future jointly from one long-lengthscale GP | Saved configuration, generator and kernel parameters; example nonconstant windows |
| 1 | Instance normalization consistent with Sundial; independent white and matched-kernel GP sources | Context-only mean/scale applied to future; independent RNG streams; source definition documented |
| 1 | Train-fitted PCA; held-out target, white-source velocity and GP-source velocity spectra | Three-row table with PC1, PC2 and cumulative PC1+PC2 energy, plus eigenvalue plot |
| 1 gate | Matched-GP velocity PC1+PC2 energy > 0.90; reject constants or absent source contrast | Gate saved before model training; target roughness/variance and white-velocity spectrum checked separately |
| 2 — 40% | AdaLN MLP and three residual blocks with S4 along horizon; same tiny encoder design, comparable capacity | Model summaries, parameter counts, initialization/training budgets and architecture provenance |
| 2 | Four source × head cells, with independent coupling in observation coordinates | Four completed configurations; same source/data family definitions; no pretrained weights |
| 2 | PC1 velocity MSE, residual PCs 3…F velocity MSE, fixed-Euler path roughness | Four-cell table of all three metrics; PC2 separately reported, raw and normalized errors distinguished |
| 2 | Generated patches plotted against target patches for every cell | Four-cell plot using shared contexts, source seeds and axis conventions; no pairing treated as unique truth |
| 2 | Prewritten 20% rule, interpreted consistently | Exact equation and numerical variance floor in protocol; rule evaluated without changing it after seeing results |
| 3 — 30% | Mandatory third head on matched-GP source, same encoder; whitened MLP or low-dimensional readout | Control row with same three metrics, trained and evaluated even if no raw-MLP gap appears |
| 3 | Training-velocity PCA whitening; evaluate predictions in original coordinates | Saved transform/eigenvalue floor, inverse-transform test, observation-space metric computation |
| 3 | One-paragraph evidence-based recommendation | Keep MLP, investigate S4, or coordinate change, with uncertainty and scope limited to this synthetic geometry |
| Optional | Real smooth family only if spectrum gate passes | Omission explicitly stated, or separate gate and held-out results; not needed for completion |
| Cross-cutting | Independent train/validation/test synthetic draws; tests for metrics | Split seeds/checksums, no test-dependent tuning, dummy-tensor projection and roughness tests |
| Cross-cutting | Reproducible execution and honest MPS use | Saved configuration, seeds, device, software versions, logs, checkpoints, upstream licenses; explicit CPU fallbacks |

## Decisions that must be fixed before scored runs

**Normalization and independence.** Compute the centering and scale from the observed lookback, not the horizon. Specify whether the GP source is drawn directly in normalized coordinates or receives a context-dependent affine transform. Either can be independent of the target conditional on the context, but they produce different source distributions. Do not claim identical source/target normalized covariances merely because their unnormalized kernels match. Record the source scale; do not tune it on test performance.

**Encoder fairness.** “Same encoder” must be operationalized. A shared architecture, same initial parameters and equal training schedule controls design, but independently trained copies end with different representations. Say so. A literally shared frozen encoder is a stronger control but changes training. Never silently train a separate privileged lookback predictor only for one head.

**PC metrics.** For velocity error `e = v_theta - (y - epsilon)`, train-PCA basis `q_j` and train eigenvalues `lambda_j`, retain the assignment's raw quantities:

- `M1 = mean((e @ q_1)^2)`.
- `MR = mean_i mean_{j=3..F} ((e_i @ q_j)^2)` (mean per retained component, not a sum).
- `R = mean_i mean_{k=1..F-1} |x_i,k+1 - x_i,k|` in normalized observation coordinates.

Report PC2 separately so it does not disappear between common and residual metrics. Source-specific PCA bases are appropriate within-source head comparisons; comparing across sources also changes the target velocity distribution. Add variance-normalized component errors `Nj = Mj / max(lambda_j, delta)` and their residual mean. They diagnose relative accuracy in low-energy directions, whereas raw errors preserve the assignment's literal units. Publish `delta` and label all variants clearly. Never infer high-frequency jitter from a large residual-PC error alone; PCs are data-dependent directions, and roughness directly checks temporal variation.

**The 20% rule is underspecified in the assignment.** Specify whether “within 20%” means a two-sided interval `[0.8, 1.2]` for residual/common error ratio or the one-sided condition `MR <= 1.2 M1`; the former perversely rejects a much smaller residual error. Preserve a literal two-sided diagnostic if using a one-sided operational criterion. State whether raw or variance-normalized errors determine each flag; report both rather than substituting one silently. Roughness agreement is naturally two-sided: `abs(R/R_target - 1) <= 0.20`. Tiny denominators require an explicit convention. A criterion flag is not by itself evidence of good conditional forecasting.

**Regression noise and oracle.** A random future is not uniquely identified by the lookback. For independent pairing, the optimal vector field is `E[y - epsilon | x_t, t, context]`; sample-pair velocity MSE cannot generally vanish. A known-GP conditional oracle distinguishes irreducible noise from approximation error. Conditional on the full observed lookback, let target mean/covariance be `(mu_y, C_y)` and independent source `(mu_s, C_s)`. Then

`A_t = t² C_y + (1-t)² C_s`, `B_t = t C_y - (1-t) C_s`,

`v*(x,t) = mu_y - mu_s + B_t A_t^{-1} [x - t mu_y - (1-t) mu_s]`.

Its conditional error covariance is `C_y + C_s - B_t A_t^{-1} B_t.T`. Apply the actual context-specific normalization in this derivation, and use stable solves with a declared jitter. This oracle uses more information than a compressed encoder and should be described as a lower-bound diagnostic. An oracle with poor Euler roughness indicates a discretization/geometry problem; add a larger-step-count convergence check without replacing the preregistered score. A zero-velocity baseline is also informative because it exposes whether the GP source already resembles the target marginal. It need not be a valid conditional forecast.

**Whitening.** For a constant linear transform `W` fitted only to training velocities, transform both `x_t` and its derivative: `z_t = W x_t`, `dz/dt = W(y-epsilon)`. Map predictions and samples back using the inverse before raw metrics. If centering positions, use one fixed position offset; a derivative does not inherit that offset. Full-rank whitening preserves all dimensions and changes optimization/loss weighting; two-PC truncation removes directions and is a different intervention. Floor small eigenvalues to avoid amplification of numerical noise, report the floor, and test round trips.

**Evidence and conclusions.** Multiple seeds, paired evaluation draws and equal update budgets reduce noise and confounding. A small synthetic study can support a design choice; it cannot prove that a head's Jacobian is the mechanism. If all four cells fail, first examine oracle/floor, geometry, discretization and optimization, then rerun a new declared geometry if needed. Do not retrofit the success definition to rescue a preferred result. Optional real-series work is lower priority than a complete, validated synthetic study.

## Poster and oral deliverables

The poster rubric asks for a common research problem, synthesis of assumptions/methods, reconstructed published evaluation protocols, claims versus evidence, missing comparisons, and a feasible project question/plan. The existing proposal poster is not a substitute for completed experimental tables and a report. A completed-project presentation should identify measured results and remaining limitations. All team members must present and answer questions; this cannot be verified by code or an artifact.

## Completion record — 2026-10-04

| Task | Completed evidence |
|---|---|
| Task 1 | `output/pdf/related-work.pdf` (one page); `output/analysis/geometry_table.csv`; `geometry_spectra.pdf`; both original PDFs preserved in `references/` |
| Task 2 | 20 primary source×head runs (4 cells×5 seeds), plus required three-metric table and per-cell paths in `output/analysis/`; report pages 3–8 |
| Task 3 | Five primary GP-whitened MLP runs; same original-coordinate metrics; recommendation in `docs/conclusions.md` and report page 11 |
| All-fail branch | Fresh lengthscale-6 geometry gate and 15 separate trained runs; `output/analysis-lengthscale6/`; report page 9. Formal method conclusion remains withheld |
| Additional checks | Nine lower-rate MPS warm restarts; fixed-checkpoint CPU Euler checks; analytical GP diagnostic; geometry sensitivity scan; report pages 8–10 |
| Verification | 26 passing tests with MPS access; `scripts/check_delivery.py` recomputes all 49 saved-run metrics and checks selected checkpoints, hashes and report completeness |
| Handoff | `README.md`, `docs/README_RU.md`, exact dependencies, configurations, checkpoints, samples and a checksummed project archive |

The optional real-series extension was omitted in the original 4 October delivery. The existing LaTeX poster describes the proposal, while `output/pdf/research-report.pdf` presents measured results. Oral participation by every team member is a human delivery requirement and is not claimed as completed by the code.

## Separate extension completed — 2026-10-08

`configs/extension.json` and `docs/extension_protocol.md` define 96 new MPS fits:
dense/sparse MLP, S4 and U-Net; temporally white/GP sources; three seeds; correlated
three-channel GP, ETTh1 OT-only, seven-channel ETTh1 and eight-channel Exchange.
Real targets, full joint-channel forecasting, train-only global channel scaling,
chronological contained windows, proper forecast scores and simple baselines are
implemented and independently verified. All original numerical files and primary
results are preserved.

This extension reports joint E2/E2C and per-channel geometry, but does not impose
or redefine the original univariate rank-two gate. It is a generalization study
with changed normalization, horizons and budgets, not a substitute for Tasks 1–3.
ETTh1 OT happens to have GP-velocity validation E2=0.9076; the other extensions
must not be described as automatic additional passes of the original stress test.
See `docs/extension_findings.md` and `output/pdf/multivariate-real-extension.pdf`.
