# Kernel and window-to-lengthscale scan: protocol

Status: written on 2026-10-12, before any scored training or test-set inspection for this scan. The configuration (`configs/kernel_scan.json`, including its `analysis` thresholds) and the SHA-256 hashes of every numerical source file are frozen into `results/kernel-scan/manifest.json` on the first invocation. A resumed run refuses any change.

This is a **supplementary, exploratory** study. It does not rewrite any Task 1–3 result, pass flag or conclusion of an earlier study. The pre-training geometry in §3 was computed from this configuration's training and validation draws; it uses no model and no test window.

## 1. Question and motivation

**Question.** How does the AdaLN-MLP vs S4 gap, and the whitened-MLP control, depend on the GP process? The scan varies kernel roughness and the window-to-lengthscale ratio F/ℓ.

**The existing evidence covers one point.** Every completed univariate run uses an RBF kernel at F = 16, with F/ℓ = 2 (main, controlled follow-up, covariance mixture) or 2.7 (lengthscale-6). At that point:

- MLP has lower raw residual-PC error than S4: 0.049 vs 0.058 in main, worse for S4 in 5/5 seeds.
- On a pure GP source, S4 has slightly lower CRPS: 0.487 vs 0.502 in the mixture study, same direction on both datasets.

**Geometry depends mainly on F/ℓ.** For a stationary kernel on a unit grid, the window spectrum is set mainly by F/ℓ.

- RBF velocities are near rank 2 only at small F/ℓ.
- OU and Matérn spectra decay as power laws, not super-exponentially. They have many signal-bearing small directions.
- The table in §3 shows that every kernel fails the 0.9 gate at F/ℓ ≥ 4, and OU fails it at every ratio. Matérn-5/2 and Matérn-3/2 pass only at F/ℓ = 2.

**Why Gauss–Markov kernels matter here.** A Matérn kernel with ν = p + ½ is a stationary Gauss–Markov process with a (p+1)-dimensional linear SDE state: OU for p = 0, Matérn-3/2 for p = 1, Matérn-5/2 for p = 2. The RBF has no finite-order state-space form. The latent Gauss–Markov prior programme, which the assignment's design constraint targets, therefore lives in the family that no completed study has tested.

**Two mechanisms make opposite predictions.**

- **H_collinear** is the assignment's original mechanism. The near rank-2 velocity `y−ε` lets common-mode gradients dominate an unstructured head. This predicts that S4's relative advantage is **largest** where the GP velocity is most collinear (RBF at F/ℓ = 2) and shrinks as F/ℓ grows or the kernel roughens. The anchor results already contradict its level prediction.
- **H_temporal** is the "temporal law" reading in the assignment's Task-3 branch. When the target has many signal-bearing, locally structured directions, a head that convolves along the patch with F-independent parameters has the right inductive bias, while an MLP must learn each direction through F-dimensional maps. This predicts that S4's relative advantage **grows** with F/ℓ and with roughness.

**Pre-declared direction:** H_temporal. The direction of the trend therefore also discriminates between the two mechanisms.

## 2. Design

### 2.1 Kernels

The scan uses four kernels, all with variance 1 and the main study's nugget of 10⁻⁶ on the diagonal. Lag r is in grid steps.

| Kernel | k(r)/σ² |
|---|---|
| RBF | exp(−r²/2ℓ²) |
| Matérn-5/2 | (1 + √5 r/ℓ + 5r²/3ℓ²) exp(−√5 r/ℓ) |
| Matérn-3/2 | (1 + √3 r/ℓ) exp(−√3 r/ℓ) |
| OU (Matérn-½) | exp(−r/ℓ) |

- **Parametrization.** This is the Rasmussen–Williams parametrization: Matérn-ν tends to the RBF with the same ℓ as ν → ∞. F/ℓ therefore uses the same ℓ for every kernel.
- **Roughness order.** At equal ℓ the order is RBF < Matérn-5/2 < Matérn-3/2 < OU. The lag-one correlation decreases in that order, and a test checks it.
- **RBF identity.** The RBF matrix is bit-identical to `data.rbf_covariance`, and RBF windows are bit-identical to `data.sample_dataset`.

### 2.2 Geometries: two realizations of F/ℓ ∈ {2, 4, 8}

| id | F | lookback | ℓ | F/ℓ | axes |
|---|---|---|---|---|---|
| F16_l8 | 16 | 32 | 8 | 2 | horizon, lengthscale (anchor = main's geometry) |
| F32_l8 | 32 | 64 | 8 | 4 | horizon |
| F64_l8 | 64 | 128 | 8 | 8 | horizon |
| F16_l4 | 16 | 32 | 4 | 4 | lengthscale |
| F16_l2 | 16 | 32 | 2 | 8 | lengthscale |

**The horizon axis is primary.** It varies F at fixed ℓ = 8, with lookback 2F as in main's 32/16.

- The anchor is exactly main's geometry.
- Grid resolution stays at 8 samples per lengthscale, so each kernel's per-step correlation, per-step roughness and nugget position stay fixed. Only the number of lengthscales per window changes.
- The assignment itself names "lengthen the patch" as the remedy for an empty task, alongside shortening the lengthscale.
- S4 is a sequence model. Any advantage from its temporal law needs a longer patch to show.

**The horizon axis has a confound.** F also changes the MLP's input and output dimension, the S4 convolution length and the encoder input.

**The lengthscale axis removes that confound.** It holds F = 16 and L = 32, so every architectural dimension is identical to main, and changes only the process. Its cost is a coarser grid: at ℓ = 2 there are two samples per lengthscale. The two realizations of each ratio have nearly identical velocity geometry (§3; RBF at ratio 8: GP top-2 0.573–0.579 vs 0.570–0.584). A disagreement between axes can therefore be attributed to F, i.e. dimension and sequence length, rather than to velocity collinearity.

**Not scanned:** F = 128, the lengthscale axis at F ≥ 32, and nugget or variance changes. The reasons are compute and keeping one factor per axis.

### 2.3 Sources, heads and widths

**Sources.**

- **White:** N(0, I_F).
- **Matched GP:** the raw kernel matrix on the horizon grid, including the nugget, drawn in normalized coordinates independently of the target. This is main's convention and its known caveat applies: the source is narrower than the normalized target (`docs/final_report_addendum.md` §8).

**Heads.** These are main's three: AdaLN MLP, three-block S4-NPLR, and PCA-whitened MLP (`experiment.CoordinateModel`).

- The whitened MLP uses the training-velocity PCA of its own source, whitened input and output coordinates, and loss in whitened coordinates.
- The variance floor is 10⁻⁶ of the leading eigenvalue, as in main.
- All six source × head cells run at every kernel and geometry. The whitened MLP on the white source is a control; prior_mixture also ran it.

**Widths and parameter counts.** Encoder dim 32, depth 3, S4 state 16 and time dim 32 are as in main.

| F | MLP width | S4 width | MLP params | S4 params | S4/MLP − 1 |
|---|---|---|---|---|---|
| 16 | 56 | 56 | 65,464 | 65,281 | −0.28% |
| 32 | 56 | 57 | 68,296 | 68,174 | −0.18% |
| 64 | 56 | 59 | 73,960 | 74,038 | +0.11% |

- **Why the MLP width is fixed.** The MLP stays at main's width 56 at every F. It is the Sundial-style head under test, and only its F-dimensional input and output maps change.
- **Why the S4 width is raised.** S4's parameter count is independent of F except through the shared encoder. At width 56 it would be 3% smaller than the MLP at F = 32 and 8% smaller at F = 64. Its width is therefore set per F to match total trainable parameters to within 0.3%.
- **Disclosed bias.** Matching gives S4 1.8% and 5.4% more hidden width exactly where H_temporal predicts that S4 wins. Shrinking the MLP to 55 or 53 instead would bias in the same direction. Keeping width 56 for both would instead favour the MLP by 3% and 8% in parameters.
- **Verification.** `create_model` and `S4Velocity` support F ∈ {32, 64}: the finite-horizon NPLR kernel is built by Krylov doubling for any length. Tests check finite forward and backward passes and the kernel shape at both F.

### 2.4 Replication, seeds and pairing

There are three **independent data draws**. Each has its own optimization seed.

| Replicate | data_seed | seed |
|---|---|---|
| 1 | 307041 | 50 |
| 2 | 407041 | 51 |
| 3 | 507041 | 52 |

Offsets from data_seed:

| Offset | Draw |
|---|---|
| +0 / +1 / +2 | train / validation / test windows |
| +100 / +110 | source draws for the training PCA / validation spectrum |
| +800 / +900 | fixed validation / test velocity pairs (+1 for their times) |
| +1000 | ensemble sources |
| +1100 / +1200 | exact oracle paths / oracle ensemble |

None of these collide with an earlier study's seeds.

**Pairing within a draw.** Every head of a (kernel, geometry, source) cell shares:

- the windows, validation and test pairs, and ensemble sources;
- the initial encoder state, re-initialized from `10000+seed`;
- the training minibatch indices, normals and flow times, re-seeded from `20000+seed` after model construction. The run records hashes of the first and last training batch.

**Common random numbers.** The same standard normals are used:

- for every kernel at the same draw and window length;
- for the white and GP sources, where the GP draw is z·Cholᵀ.

Kernel and source contrasts are therefore paired. Draws are independent across replicates but **not** across kernels.

### 2.5 Training and selection

Unchanged from `configs/main.json`:

- independent coupling, linear interpolant, target velocity y − ε;
- 32,768 / 4,096 / 8,192 windows;
- 6,000 updates, batch 256;
- AdamW at a constant learning rate of 10⁻³, weight decay 10⁻⁴, with official S4's no-decay rule for SSM parameters;
- gradient clipping at 1;
- Euler with 64 steps.

The checkpoint is the best **validation objective** among evaluations every 500 updates. That objective is raw MSE for MLP and S4, and whitened-coordinate MSE for the whitened MLP. There is no tuning, no early stopping, no schedule change and no extension.

Test windows are generated only after the last update and checkpoint selection, and a test checks this. A run that raises on a non-finite validation loss is reported as failed and counted as the worst outcome for that head in that cell. It is not re-run with changes.

## 3. Geometry record (Task-1 analogue; never a gate)

For every kernel × geometry × draw, `geometry.json` records:

- the raw-kernel spectrum;
- the training-target PCA and the training-velocity PCA per source, with all eigenvalues;
- validation top-2 energy, effective rank, and the number of PCs needed for 90% and 99% of energy;
- the floored-direction count of the whitening floor;
- the nugget floors, the eigenvalue-to-floor ratio of every PC, the signal count m and the PC groups;
- the assignment's Task-1 branch label;
- the reference outcome of main's gate. The gate is not enforced.

Values across the three draws:

| Kernel | Geometry | F/ℓ | raw-kernel top-2 | GP velocity top-2 | white velocity top-2 | m | tail PCs | Task-1 branch |
|---|---|---|---|---|---|---|---|---|
| RBF | F16_l8 | 2 | 0.971 | 0.976–0.978 | 0.840–0.854 | 7 | 8–16 | contrast exists |
| RBF | F32_l8 | 4 | 0.824 | 0.849–0.855 | 0.637–0.651 | 11 | 12–32 | both full rank |
| RBF | F64_l8 | 8 | 0.543 | 0.573–0.579 | 0.379–0.388 | 18 | 19–64 | both full rank |
| RBF | F16_l4 | 4 | 0.825 | 0.852–0.855 | 0.658–0.664 | 10 | 11–16 | both full rank |
| RBF | F16_l2 | 8 | 0.544 | 0.570–0.584 | 0.415–0.428 | 15 | 16 | both full rank |
| Matérn-5/2 | F16_l8 | 2 | 0.939 | 0.948–0.949 | 0.792–0.798 | 16 | none | contrast exists |
| Matérn-5/2 | F32_l8 | 4 | 0.771 | 0.802–0.805 | 0.588–0.596 | 32 | none | both full rank |
| Matérn-5/2 | F64_l8 | 8 | 0.509 | 0.540–0.544 | 0.352–0.359 | 64 | none | both full rank |
| Matérn-5/2 | F16_l4 | 4 | 0.772 | 0.798–0.806 | 0.605–0.617 | 16 | none | both full rank |
| Matérn-5/2 | F16_l2 | 8 | 0.509 | 0.537–0.548 | 0.390–0.401 | 16 | none | both full rank |
| Matérn-3/2 | F16_l8 | 2 | 0.911 | 0.923–0.924 | 0.755–0.761 | 16 | none | contrast exists |
| Matérn-3/2 | F32_l8 | 4 | 0.738 | 0.770–0.773 | 0.558–0.564 | 32 | none | both full rank |
| Matérn-3/2 | F64_l8 | 8 | 0.488 | 0.518–0.521 | 0.336–0.341 | 64 | none | both full rank |
| Matérn-3/2 | F16_l4 | 4 | 0.739 | 0.765–0.774 | 0.573–0.587 | 16 | none | both full rank |
| Matérn-3/2 | F16_l2 | 8 | 0.489 | 0.517–0.526 | 0.376–0.386 | 16 | none | both full rank |
| OU | F16_l8 | 2 | 0.773 | 0.789–0.794 | 0.600–0.609 | 16 | none | both full rank |
| OU | F32_l8 | 4 | 0.606 | 0.640–0.646 | 0.445–0.448 | 32 | none | both full rank |
| OU | F64_l8 | 8 | 0.405 | 0.433–0.438 | 0.277–0.279 | 64 | none | both full rank |
| OU | F16_l4 | 4 | 0.610 | 0.639–0.647 | 0.467–0.477 | 16 | none | both full rank |
| OU | F16_l2 | 8 | 0.415 | 0.444–0.450 | 0.325–0.332 | 16 | none | both full rank |

Only three cells have the assignment's numerical contrast: RBF, Matérn-5/2 and Matérn-3/2 at F/ℓ = 2. OU never does. That OU never shows the contrast is itself a finding: the near-rank-2 premise of the "head, not prior" cell does not hold for the simplest Gauss–Markov prior at any tested ratio.

### 3.1 Residual groups per F

- **Literal residual (co-primary E1):** velocity PCs 3..F at every F. This is the assignment's definition, `metrics.velocity_metrics`' `residual_mse`, and the input to the literal rule.
- **Nugget floor.** The raw nugget is white and independent of the history statistics. It therefore adds exactly `nugget·E_train[1/s²]` to every normalized-target eigenvalue, where s is the Sundial history scale. The isotropic velocity floor is that value plus 1 (white source) or plus the nugget (GP source).
- **Signal count.** m is the number of training-target PCs with eigenvalue ≥ 2 × the target floor, i.e. signal variance at least equal to the floor.
- **Signal residual:** velocity PCs 3..m.
- **Tail:** velocity PCs m+1..F.
- **Scope of m.** The same m applies to both sources and all heads of a kernel × geometry × draw, and it is identical across the three draws for every cell.

Resulting groups:

- **RBF tails:** PCs 8–16, 12–32 and 19–64 on the horizon axis; 11–16 and 16 on the lengthscale axis.
- **Matérn and OU:** no tail at any geometry. Their smallest target eigenvalue is at least about 20 times the floor. For Gauss–Markov kernels the literal residual is all signal.
- **Difference from the addendum.** At the anchor, PC7 carries about 3.9 times the floor in every draw, so it counts as signal. The post-hoc addendum grouped PCs 7–16 with the nugget.

Tail errors measure jitter injected where the target has none (addendum §3). Signal-residual errors measure how well the head learns small structured directions.

## 4. Metrics

All metrics are in normalized observation coordinates ℝ^F, on the cell's held-out test draw.

**Velocity (8,192 fixed test pairs)**, from the unchanged `metrics.evaluate`:

- per-PC raw MSE in the cell's training-velocity basis (errors not centred), with PC1, PC2 and the literal residual mean;
- floor-normalized versions;
- group means for the signal residual and the tail;
- per-PC MSE in the fixed training-target basis, for cross-source comparisons.

**Paths and rules:**

- generated vs target roughness on 1,024 paths with Euler 64;
- the literal two-sided 20% rule, unchanged;
- a diagnostic copy of the rule with the signal residual in place of the literal residual.

**Forecast:** `ensemble_evaluation.context_scores`, as prior_mixture uses it, on the first 256 test histories × 32 source draws shared by all heads. It reports fair CRPS, fair energy score, mean-forecast MSE, and 80/90% coverage and width. Per-context arrays are saved for paired analysis.

**Privileged analytic oracle.** It is re-implemented for any kernel in `kernel_scan.py`. Tests check it against:

- `data.py` for the RBF;
- the OU Markov closed form;
- a Monte-Carlo least-squares regression on a Matérn-3/2 cell.

It is computed on the same pairs and contexts:

- oracle per-PC velocity MSE and irreducible conditional noise;
- roughness of exact conditional paths;
- exact conditional ensemble CRPS and energy score with 32 draws.

Each run records the gap to the oracle as `model − oracle` and `model / oracle` per group and for CRPS and energy. The oracle sees raw history while the networks see normalized history, so this gap is informational, not attainable.

**Solver diagnostic.** Replicate 1 only, every run: Euler 128 vs 64 on the first 16 ensemble contexts, reporting endpoint RMSE and score changes. It is not used for selection.

**Bookkeeping:** best step, curve, parameters, width, runtime, encoder hash, batch hashes and test-input hashes.

## 5. Analysis plan and decision rules

Everything below is fixed now. The thresholds are in the frozen `analysis` block of the config. All 360 runs are reported, and none is excluded.

### 5.1 Contrasts

For metric M, cell (kernel, geometry, source) and replicate r, ρ_r = M_S4 / M_MLP. The cell summary ρ̄ is the geometric mean over replicates. Whitened-MLP ratios against MLP and against S4 are defined the same way.

**Endpoints.**

- **Co-primary:** E1, the literal residual-PC raw MSE, is the mechanism endpoint. E2, the fair CRPS, is the forecast endpoint. The verdicts below are given separately for E1 and E2. Disagreement between them is reported as such, as in the earlier studies.
- **Secondary:** PC1 MSE, signal residual, tail, roughness distance |generated/target − 1|, energy score, oracle gaps, coverage, the literal rule and the signal-group rule.

**Cell-level classification:**

| Label | Condition |
|---|---|
| S4 advantage | ρ_r < 1 in every replicate **and** ρ̄ ≤ 0.8 for velocity-MSE metrics (the assignment's own 20% tolerance), ≤ 0.95 for CRPS |
| MLP advantage | ρ_r > 1 in every replicate and ρ̄ ≥ 1/0.8 (velocity MSE) or ≥ 1/0.95 (CRPS) |
| Match | Anything else; "direction-consistent" is reported separately |

The 0.95 CRPS threshold reflects that CRPS includes predictive spread common to both heads. Under it, the earlier 3% GP-source CRPS difference would be direction-consistent but not material.

### 5.2 Trend and kernel tests (GP source)

- **Axis trend.** For each kernel, axis (horizon: F16_l8 → F32_l8 → F64_l8; lengthscale: F16_l8 → F16_l4 → F16_l2) and replicate, take the OLS slope b_r of log ρ_r on log₂(F/ℓ) over the three ratios. The trend favours S4 if b_r < 0 in every replicate. It favours the MLP (reversed) if b_r > 0 in every replicate. Otherwise it is flat or inconsistent.
- **Kernel contrast.** Per replicate, take log ρ(OU) − log ρ(RBF) at each geometry, and its mean over the five geometries. It favours S4 if the five-geometry mean is negative in every replicate. The ordering over all four kernels is reported descriptively.

### 5.3 Verdict on H_temporal (separately for E1 and E2, GP source)

| Verdict | Condition |
|---|---|
| Supported | (a) the horizon-axis trend favours S4 for ≥ 3 of 4 kernels, **and** (b) the OU − RBF contrast favours S4, **and** (c) at least one cell with F/ℓ ≥ 4 has an S4 advantage |
| Partial | At least one of (a)–(c) holds, but not all three. For example, the gap narrows without S4 becoming materially better, or the effect appears without a consistent trend |
| Reversed | The horizon-axis trend favours the MLP for ≥ 3 kernels **and** the OU − RBF contrast favours the MLP. This is the direction H_collinear predicts, though S4 does not lead at the anchor |
| Refuted | None of (a)–(c) holds and no cell anywhere has an S4 advantage on that endpoint |

**Axis attribution:**

- Both axes favour S4 for the same kernels: an F/ℓ (process-geometry) effect.
- Only the horizon axis: a window-length or dimension effect, not velocity collinearity.
- Only the lengthscale axis: a per-step roughness or resolution effect.

### 5.4 Mapping to the assignment's conclusion branches

Each kernel × geometry receives three labels. No new branch is invented.

1. **Task-1 label** (§3): contrast exists / both full rank / both collapsed / collapsed targets.
2. **Literal Task-2 label.** If every head × source fails the literal rule in every replicate, the label is **Empty task**. Otherwise it is the Task-2 branch whose literal "works" conditions hold:
   - **The source is the knob:** the MLP works with one source and not the other, in at least 2 of 3 replicates.
   - **Head, not prior** or **S4 was doing the GP's job:** the corresponding works patterns.
3. **Head-comparison label.** On the assignment's three numbers (PC1 MSE, residual MSE, roughness distance), using §5.1:
   - **Head, not prior:** GP source, MLP matches S4 on PC1, and S4 has an advantage on residual or roughness distance.
   - **Prior was overstated:** GP source, MLP matches or beats S4 on all three numbers.
   - **S4 was doing the GP's job:** S4 has an advantage under the white source as well, and S4-GP CRPS is not materially (5%) better than S4-white CRPS.
   - **None:** anything else (e.g. S4 wins on PC1).

**Precedence when labels conflict.** The addendum notes that "Empty task" and "Prior was overstated" can hold together, and the assignment does not order them. Both labels are therefore reported, and the literal label is listed first.

**Scope of an assignment-literal reading.** Such a reading is made only in cells labelled "contrast exists": RBF, Matérn-5/2 and Matérn-3/2 at F16_l8. The assignment's Task-1 rule forbids interpreting Task 2 in full-rank cells. Those cells are scan points describing regime dependence, and they enter the recommendation only through the explicitly extended rule below.

**Task 3** (GP source; whitened MLP vs S4 vs MLP), per endpoint, in cells where S4 has an advantage over the MLP:

| Branch | Condition |
|---|---|
| Whitening closes the gap | The whitened MLP matches or beats S4 (ρ̄ whitened/S4 ≤ 1.25 for velocity MSE, ≤ 1/0.95 for CRPS) |
| Third head does not close the gap, S4 still wins | S4 has an advantage over the whitened MLP as well |

In cells without an S4 advantage:

| Branch | Condition |
|---|---|
| Third head and S4 tie raw MLP | Neither S4 nor the whitened MLP has an advantage over the MLP |
| Whitening branch | The whitened MLP alone has an advantage |

**Recommendation for the latent Gauss–Markov programme.** This is an extension beyond the assignment's gate, declared now. It is based on the six Gauss–Markov cells with F/ℓ ≥ 4 on the horizon axis (Matérn-5/2, Matérn-3/2 and OU at F32_l8 and F64_l8), GP source, both endpoints:

| Recommendation | Condition |
|---|---|
| Switch to an S4/SSM-like head | "S4 still wins" holds in ≥ 4 of the 6 cells on E1 or E2, without the other endpoint showing an MLP advantage in those cells |
| Flow-match after whitening | "Whitening closes the gap" holds in ≥ 4 of the cells where S4 has an advantage, or the whitened MLP has an advantage over both other heads in ≥ 4 of 6 |
| Keep TimeFlow's head | No S4 or whitened-MLP advantage in ≥ 4 of 6 |
| No change to the recommendation | Anything else. The regime dependence is reported instead |

## 6. Interpretation limits

- **Three draws.** "Same sign in all three replicates" has null probability ¼ per cell, and there are many cells. Cell-level labels are descriptive. The verdicts rely on pooled trend and kernel patterns, not on single cells.
- **Fixed budget.** There are 6,000 updates and convergence is not established. Each run reports whether the selected checkpoint is at the last update. A larger F may need more updates. That would be a budget effect, and it is reported, not corrected.
- **Matched-parameter widths** introduce the small S4 capacity bias stated in §2.3.
- **Common random numbers** pair kernels, so kernel contrasts are not independent replications.
- **Privileged oracle.** It sees raw history while the networks see normalized history.
- **Source narrower than target.** The GP source is narrower than the normalized target (main's convention).
- **Literal rule limitations.** Two known issues remain. Nugget directions dilute the residual mean, which the tail and signal split addresses. The roughness band cannot detect missing curvature (addendum §4).
- **CRPS across F.** Absolute CRPS is not comparable across F. Use within-geometry ratios or ratios to the oracle.
- **Scope.** A fitted head describes this synthetic cell. Nothing here is a claim about pretrained Sundial or full-scale TSFlow.

## 7. Compute estimate

**Calibration.** The reference timing supplied with the request: a 6,000-step fit at F = 16 takes about 0.5 min (MLP) and 1.3 min (S4) on an M3 Pro with MPS. The scaling with F comes from MPS micro-benchmarks on this machine (300–800 steps, batch 256, timing only, run under load, about 1.6× slower in absolute terms). They gave per-step time relative to F = 16 of:

- S4: about 1.4× at F = 32 and about 2.8× at F = 64;
- MLP and whitened MLP: flat within noise.

**Evaluation per run** (8,192 velocity pairs, 1,024 paths, 256 × 32 ensemble paths, CPU oracle about 1 s) adds about 10 / 20 / 35 s for S4 at F = 16 / 32 / 64, and a few seconds for the MLPs.

| Block (per replicate) | Fits | Estimated time |
|---|---|---|
| F16_l8 (anchor) | 24 | ~20 min |
| F32_l8 | 24 | ~26 min |
| F64_l8 | 24 | ~42 min |
| F16_l4 + F16_l2 | 48 | ~39 min |
| **One replicate** | **120** | **~2.1 h** |

| Scope | Fits | Estimated time |
|---|---|---|
| Horizon axis, 3 replicates | 216 | ~4.4 h |
| Lengthscale-axis additions, 3 replicates | 144 | ~2.0 h |
| **Full scan** | **360** | **~6.3 h** (about 10 h if the machine is as loaded as during the benchmark) |

S4 fits take about two thirds of the total. Expected output is about 1.1 GB, mostly prediction and ensemble arrays and roughly 360 checkpoints of about 0.3 MB each.

**Measured CUDA timing (2026-10-12, before any scored run).** The host has an RTX 2060 and a Ryzen 5 2600, with torch 2.14.1+cu130. The benchmark was timing only: 300 steps at batch 256, plus one evaluation pass.

| Head | Per step at F = 16 / 32 / 64 | Per fit (6,000 steps plus evaluation) |
|---|---|---|
| MLP | 7.4 / 10.8 / 5.5 ms | 0.75 / 1.1 / 0.6 min |
| S4 | 27.8 / 29.2 / 26.9 ms | 2.9 / 3.0 / 2.9 min |

Time is nearly flat in F because the bound is CPU and kernel launches, not the GPU. Three concurrent processes ran at almost solo speed: S4 at 29–30 ms per step and MLP at 8–9 ms.

- One process: about 3.0–3.3 h per replicate, about 9–11 h for the full scan.
- Three processes, one per data draw: about 3.5–4 h for the full scan.

## 8. Commands (sequential MPS; nothing else on the GPU)

```sh
.venv/bin/python -m pytest tests/test_kernel_scan.py -q
.venv/bin/python -m fm_stress.kernel_scan --prepare-only
.venv/bin/python -m fm_stress.kernel_scan --device mps 2>&1 | tee -a results/kernel-scan-training.log
```

Equivalent staged form, horizon axis first:

```sh
.venv/bin/python -m fm_stress.kernel_scan --geometry F16_l8,F32_l8,F64_l8 2>&1 | tee -a results/kernel-scan-training.log
.venv/bin/python -m fm_stress.kernel_scan --geometry F16_l4,F16_l2 2>&1 | tee -a results/kernel-scan-training.log
```

**Run order.** Runs go replicate first, then geometry, then kernel, then source, then head. An interruption therefore leaves complete replicates.

**Resuming.** Re-running the same command skips completed runs after re-hashing their checkpoint and samples. A partially trained run restarts from scratch.

**Filters and status.**

- `--kernel`, `--ratio`, `--geometry`, `--head`, `--source` and `--data-seed` take comma-separated values for partial runs.
- `--list` prints each selected run's status without training.

**Device.** `--device mps` raises if MPS is hidden and never falls back to CPU. Use `--device cpu` explicitly only for development output directories.

**Device amendment (2026-10-12, before any scored run).** The scored scan runs on a Linux host with an NVIDIA RTX 2060 (6 GB), using `--device cuda`. CUDA follows the same rule as MPS: it raises if unavailable and never falls back to CPU. GPU floating point differs between MPS and CUDA, so one results directory holds runs from one device only, and the runner refuses runs from a second device. Grid, seeds, thresholds and decision rules are unchanged. The §7 estimates were calibrated on Apple MPS; the measured CUDA timing is added in §7.

**Parallel execution amendment (same date, before any scored run).** The scan runs as three concurrent CUDA processes, one per data draw, via `--data-seed`. Group directories are `<kernel>/<geometry>/draw<seed>`, so the processes write disjoint files. The shared root (manifest and geometry records) is created first by a single `--prepare-only` call. Each process is an independent computation: concurrency changes timing only, never the data, seeds or arithmetic of a run.

## 9. Provenance and remaining steps

- **Manifest.** `results/kernel-scan/manifest.json` records the config hash and the SHA-256 of `kernel_scan.py`, `models.py`, `data.py`, `metrics.py`, `experiment.py`, `ensemble_evaluation.py` and `robustness.py`. Any change refuses with "Changed provenance: use fresh output directory".
- **Geometry and oracle.** `geometry.json` is recomputed and compared on every invocation. The per-group `oracle_{source}.json` is verified against the hashes of the test inputs.
- **Development data.** `configs/kernel_scan_smoke.json` and any CPU output are development data only, never evidence.
- **Before any report.** An independent NumPy audit (`scripts/check_kernel_scan.py`, with no `fm_stress` data, PCA or metric imports) and an analysis module (`fm_stress/analyze_kernel_scan.py`) implementing §5 must be written. The audit must pass before the analysis is reported. Neither is part of this deliverable.
