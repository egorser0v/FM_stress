# Sparse representations and joint whitening: measured findings

Completed on 8 October 2026: **288 new MPS fits** (264 main + 24 structural controls), with 96 frozen reference fits. Every neural variant was evaluated with both temporally white and GP sources and three optimization seeds. The source is the random starting trajectory: temporally white noise or correlated GP curves; this is distinct from the target dataset. All original experiments remain unchanged. The 122-test suite and independent run, metric, source-hash, CPU-replay and aggregate/bootstrap audits passed.

## Whitening is useful in some geometries, not a general improvement

Fair CRPS below is averaged over three seeds in TRAIN-standardized observation coordinates; lower is better. Whitening retains all joint time/channel coordinates. The balanced-loss MLP uses the same variance-weighted objective but keeps the original channel-standardized input/output coordinates.

| Dataset | Source | MLP | Whitened MLP | Balanced-loss MLP | Whitening change |
|---|---|---:|---:|---:|---:|
| GP, 3 channels | white | 0.32877 | 0.34386 | 0.38197 | +4.6% |
| GP, 3 channels | gp | 0.33282 | 0.30948 | 0.61569 | -7.0% |
| ETTh1 OT, 1 channel | white | 0.14222 | 0.12839 | 0.12646 | -9.7% |
| ETTh1 OT, 1 channel | gp | 0.19499 | 0.13209 | 0.70948 | -32.3% |
| ETTh1, 7 channels | white | 0.36212 | 0.52679 | 0.52404 | +45.5% |
| ETTh1, 7 channels | gp | 0.37503 | 0.52977 | 0.57398 | +41.3% |
| Exchange, 8 channels | white | 0.31368 | 0.70708 | 1.11382 | +125.4% |
| Exchange, 8 channels | gp | 0.38520 | 0.56404 | 1.08462 | +46.4% |

Whitening improves mean CRPS in **3/8** source/dataset cells: multivariate GP with the GP source, and ETTh1 OT with either source. It worsens both sources on real multivariate ETTh1 and Exchange. Thus multivariate whitening was implemented and tested; its real-data result is negative at this budget.

On ETTh1 OT with a GP source, whitened MLP reaches 0.13209 versus raw MLP 0.19499 and the reference S4 0.14372. This removes the previously observed advantage of S4 over raw MLP at the level of mean CRPS. It does not prove that coordinates explain every S4/MLP difference. Balanced-loss MLP is much worse with this source (0.70948), so loss reweighting alone did not reproduce the whitening benefit.

The ETTh1 OT GP whitening-minus-MLP CRPS effect is -0.06290; the paired context block-8 interval is [-0.11804, -0.01666]. These intervals condition on the fitted seeds and selected test period, and comparisons are exploratory without multiplicity correction. Raw PC1/residual MSE and forecast scores answer different questions; both are retained in the CSVs/report.

## Sparse expert routing is the most promising new sparse candidate

All four experts are small conditional velocity predictors inside flow matching. The dense and Top-2 variants share architecture, width, allocated parameter count and router regularizer. The implementation evaluates dense kernels, so these results concern forecast quality rather than sparse-kernel acceleration.

| Dataset | Source | MLP | Dense experts | Top-2 experts | Top-2 vs dense |
|---|---|---:|---:|---:|---:|
| GP, 3 channels | white | 0.32877 | 0.33779 | 0.33909 | +0.4% |
| GP, 3 channels | gp | 0.33282 | 0.33684 | 0.33506 | -0.5% |
| ETTh1 OT, 1 channel | white | 0.14222 | 0.13394 | 0.13029 | -2.7% |
| ETTh1 OT, 1 channel | gp | 0.19499 | 0.19665 | 0.18594 | -5.4% |
| ETTh1, 7 channels | white | 0.36212 | 0.37699 | 0.40462 | +7.3% |
| ETTh1, 7 channels | gp | 0.37503 | 0.35662 | 0.34007 | -4.6% |
| Exchange, 8 channels | white | 0.31368 | 0.21503 | 0.18932 | -12.0% |
| Exchange, 8 channels | gp | 0.38520 | 0.25004 | 0.23493 | -6.0% |

On Exchange, Top-2 lowers CRPS relative to dense experts by 12.0% with white and 6.0% with GP; improvements against raw MLP are approximately 39.6% and 39.0%. Block-8 context intervals for Top-2 minus dense are [-0.03433, -0.01610] and [-0.02701, -0.00305]. Both remain conditional, exploratory intervals.

On joint ETTh1, routing helps with GP (-0.01654 versus dense experts; block-8 interval [-0.02286, -0.01087]) but hurts with white (+0.02763; interval [0.02300, 0.03222]). This source-dependent contrast is more informative than a single overall model ranking.

## Autoencoders and sparsity: measured limitations

- **TopK history autoencoder:** worse mean CRPS than its dense AE counterpart in all eight source/dataset cells. It participates in FM conditioning and reconstructs only observed history. The result concerns latent width 64, k=8, auxiliary weight .05 and the fixed training budget; it is not a claim that SAEs never help.
- **Convolutional sparse history representation:** improves over its dense counterpart only on white-source Exchange (0.26658 versus approximately 0.31663). It combines learned shrinkage with an L1 penalty, so those two ingredients are not separately identified.
- **L0-gated U-Net:** all 24 selected checkpoints retain 100% nonzero deterministic gates. This was learned attenuation and stochastic regularization, not achieved channel pruning. The weak penalty/short schedule cannot support a claim that pruned U-Nets were tested successfully.
- **TopK velocity dictionary:** worse than dense coefficients in all eight main cells. A dictionary is a collection of learned velocity templates; TopK selects at most eight templates for each current input, and the selection can change during integration.

## Why the 192-template control was added

An independent structural review identified that a 64-template head cannot change some components of its starting random curve: those components survive unchanged in the forecast. Mathematically, a velocity of the form D a can move a trajectory only within the fixed learned column span of D. With 64 templates, ETTh1-7 (168 output coordinates) has at least 104 immovable directions and Exchange-8 (128 coordinates) has at least 64. Saved generated trajectories obey this invariant to about 10^-6. This restriction is shared by dense and sparse coefficients but can dominate comparisons with unrestricted networks.

The separate 24-fit follow-up increases the dictionary to 192 templates while retaining k=8, both sources and all three seeds. All selected dictionaries have full row rank (168/128), removing that architectural null space. TopK still loses to dense coefficients in all four supplemental cells, and the eight matched comparisons across 64/192 sizes all favor dense coefficients. Observed validation usage remains limited to a small subset of templates. Full-rank dictionaries do not by themselves ensure arbitrary velocities under an eight-active-template constraint.

This is an exploratory follow-up declared during the main run, not a replacement of unfavorable results. Under the 64k parameter target, predictor widths change 40→26 on ETTh1 and 43→32 on Exchange; growing the dictionary is therefore not a pure rank-only intervention.

## Keep the simple forecasting baseline in view

| Dataset | Ridge + residuals CRPS | Elastic Net + residuals CRPS | Elastic active input features |
|---|---:|---:|---:|
| GP, 3 channels | 0.19294 | 0.28457 | 25/96 |
| ETTh1 OT, 1 channel | 0.09333 | 0.09194 | 29/48 |
| ETTh1, 7 channels | 0.28702 | 0.29771 | 191/336 |
| Exchange, 8 channels | 0.06778 | 0.06764 | 70/256 |

A simple linear residual-bootstrap baseline has lower mean CRPS than every tested neural cell on each dataset. In particular, Exchange remains around 0.0676 versus 0.1893 for the strongest new neural cell. Neural ranking improvements do not close that gap. Bootstrap residuals come from in-sample TRAIN errors, so uncertainty calibration is not guaranteed.

Elastic Net chose l1_ratio=1 on all four datasets (the group-lasso endpoint), alpha .01 except .1 for ETTh1-7. Selection was solely by validation MSE among converged candidates. 45/48 fits converged; all three excluded candidates were GP alpha .001 runs reaching 50,000 iterations. Exact QR acceleration preserves the original convex problem; selected fits pass independently reconstructed primal/dual certificates.

## What the evidence supports

Whitening merits further testing on ETTh1 OT and smooth GP data, with raw-coordinate controls; it is not a default for real multivariate data. For further sparse work, the small routed-expert head has stronger evidence here than TopK autoencoders or fixed-width sparse dictionaries. The next useful test is whether those routing effects survive more training and a fresh real-data period. Do not infer universal superiority, compression speedups or a reproduced foundation-model benchmark.

125/264 main fits selected the last checked training step. A single learning rate, one sparsity setting, short horizons, three optimizer seeds and repeated use of fixed real-data periods limit generalization. Multiple seeds do not substitute for new datasets or time periods. The 64/128-step Euler checks cover only the first eight contexts of seed 31; detailed solver differences are retained.

## Artifacts and verification

- `output/pdf/sparse-whitening-extension.pdf`: readable report, main comparisons and dictionary supplement.
- `output/sparse-analysis/`: 120 main/reference aggregate cells, seed/context contrasts and nine figures.
- `output/dictionary-supplement/`: 16 cells, 120 seed contrasts and 80 context contrasts.
- `results/sparse-extension/` and `results/sparse-fullrank-dictionary/`: all 288 checkpoints, training curves, forecasts and per-run records.
- `results/diagnostics/sparse_analysis_verification.json` and `dictionary_analysis_verification.json`: independent aggregation and bootstrap arithmetic checks.
- `results/diagnostics/dictionary_span*.json`: template-rank, invariant-direction and validation-usage diagnostics.
- `scripts/run_sparse.sh`: reproduction and final audit pipeline.
