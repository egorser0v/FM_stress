# Completed extension: sparse MLP, U-Net, multivariate and real targets

96/96 declared MPS fits completed on 8 October 2026. All 58 project tests pass
with MPS available. Independent CPU recomputation verifies all 96 saved runs,
their data/transforms, checkpoints, validation scores and forecast metrics.
The original experiment's numerical source hashes remain unchanged.

## What was added

- Fixed-mask sparse AdaLN MLP (approximately 50% active connections in all linear
  maps; wider to match active capacity, dense storage/compute).
- Temporal 1-D U-Net with three resolutions and skip connections.
- Joint multichannel dense MLP and S4, not channel-independent univariate copies.
- Correlated GP targets with three channels; ETTh1 OT-only and all seven ETTh1
  variables; all eight Exchange variables as real forecasting targets.

Four heads, two source families, three seeds. All models see the full history.
Sources share channel correlation and differ in temporal covariance. Source GP
does **not** mean the target is synthetic: ETTh1 and Exchange targets are observed
data. Configuration, provenance and exact protocol: `extension_protocol.md`.

## Forecast results

Mean fair CRPS over three seeds, using 64 held-out contexts and 32 generated
futures per context. Lower is better. Units are training-standardized per channel;
compare scores within a dataset, not across different datasets.

| Dataset | Source | Dense MLP | Sparse MLP | S4 | U-Net |
|---|---|---:|---:|---:|---:|
| GP, 3 channels | Temporally white | 0.32877 | 0.32573 | 0.33299 | 0.32150 |
| GP, 3 channels | Smooth GP | 0.33282 | 0.33177 | 0.33179 | 0.33526 |
| ETTh1 OT, 1 channel | Temporally white | 0.14222 | 0.14087 | 0.17045 | 0.20451 |
| ETTh1 OT, 1 channel | Smooth GP | 0.19499 | 0.20059 | 0.14372 | 0.20015 |
| ETTh1, 7 channels | Temporally white | 0.36212 | 0.35845 | 0.40448 | 0.38252 |
| ETTh1, 7 channels | Smooth GP | 0.37503 | 0.37508 | 0.40334 | 0.43819 |
| Exchange, 8 channels | Temporally white | 0.31368 | 0.28366 | 0.33539 | 0.55622 |
| Exchange, 8 channels | Smooth GP | 0.38520 | 0.30108 | 0.42545 | 0.59271 |

Full seed SD, joint energy, mean forecast MSE, coverage/width, temporal roughness,
PCA errors and paired comparisons are in `output/extension-analysis/`.

## An observed S4/MLP crossover, with an important qualification

On univariate ETTh1 OT, S4's GP-source CRPS is about 26% lower than dense MLP's.
Under the temporally white source it is about 20% higher. The source-by-head
interaction `(S4_GP − MLP_GP) − (S4_white − MLP_white)` is −0.07950 CRPS. It is
negative in all three paired seeds. Exploratory context block-bootstrap intervals
are [−0.14359, −0.02841] for block length 4 and [−0.14937, −0.02251] for length 8.
These condition on this period and these trained models and do not adjust for the
many comparisons. They are not a population-level architecture theorem.

This dataset has validation top-two energy 0.9076 for GP-source velocity and
0.9578 for the target, using training-fitted PCA. However, **the original
small-component failure mechanism is not confirmed**: GP+S4 residual velocity
MSE is 0.03983, worse than GP+MLP's 0.03569. S4 instead has lower PC1 MSE,
6.1226 versus 10.2900. Forecast quality and small-PC error are different outcomes.

The absolute S4 forecast advantage does not repeat in seven-channel ETTh1 or
Exchange. On seven-channel ETTh1, its relative source interaction is modest and
the block-length-8 interval includes zero. On the synthetic GP and Exchange,
the S4 interaction intervals also include zero.

## Sparse MLP and U-Net

Sparse MLP is close to dense MLP on GP/ETTh1 and has a clearer descriptive gain on
Exchange: about 9.6% lower CRPS with white source and 21.8% lower with GP source.
This compares wider masked networks against narrower dense ones at approximately
equal active capacity. It does not isolate sparsity alone from width or prove a
speed advantage; the masked weights still use dense kernels.

U-Net has the lowest mean neural CRPS on the white-source synthetic GP cell, but
does not show a consistent improvement on the real datasets. More structure is
not automatically better under a short fixed budget and one common learning rate.

## Does multivariate prediction improve OT?

The OT-only rescore uses exactly the same ETTh1 test origins, OT targets and
training scaler. It compares OT output against OT output, not a seven-variable
average against one variable. Joint models have worse mean OT CRPS in seven of
eight head/source combinations. White-source U-Net is the sole small reversal
(0.19629 joint versus 0.20451 univariate), with a paired-seed interval including
zero. There is no general multivariate improvement in this experiment.

This is not evidence that other variables contain no useful information. Joint
models must predict seven outputs with a similar total active-parameter budget;
conditioning, width, source covariance and optimization all change.

## Strong baselines remain the practical reference

Ridge regression plus whole-vector training-residual resampling outperforms every
neural cell on mean CRPS **and** mean joint energy in all four datasets:

| Dataset | Ridge + residual CRPS | Lowest mean neural CRPS |
|---|---:|---:|
| GP, 3 channels | 0.19294 | 0.32150 |
| ETTh1 OT | 0.09333 | 0.14087 |
| ETTh1, 7 channels | 0.28702 | 0.35845 |
| Exchange, 8 channels | 0.06778 | 0.28366 |

The residual baseline uses in-sample training residuals and is not guaranteed to
be calibrated under distribution shift. Nevertheless, its held-out proper-score
advantage must not be hidden. The neural results are a controlled stress-test
extension, not a competitive state-of-the-art forecasting claim.

## Scope and next scientific decision

All heads received 2,000 updates at LR 0.001, selected by validation checkpoints.
This establishes performance at a fixed budget, not convergence or optimal
tuning. The new global training-channel normalization differs from the original
Sundial-style history-instance normalization. The univariate original study,
its whitening control and its declared pass flags remain a separate completed
experiment; the new tables must not be pooled with them.

Keep sparse MLP as a useful additional baseline. Use the ETTh1 OT crossover as a
specific case for further source–head investigation, while retaining the worse
S4 residual-PC result. Before claiming practical neural improvements, a future
validation-only study should examine longer budgets, head-specific tuning,
history normalization/residual parameterization, and simple linear baselines.
Those are recommendations, not experiments claimed as completed here.

## Reproduction and verification

See README commands and `configs/extension.json`. Download raw sources with
`scripts/fetch_extension_data.py`; raw data are excluded from Git and the package.
Pinned revisions and hashes are in `data/real/manifest.json`. Run the MPS suite,
then `scripts/check_extension.py` and `fm_stress.analyze_extension`.

Independent audit: `results/extension/independent_verification.json`, 96/96.
Checkpoint validation replay CPU/MPS maximum absolute MSE difference: 4.77e-7.
Full model/gradient/update device checks: 16 head×channel configurations.
Full test suite: `results/diagnostics/extension_pytest_mps.txt`, 58 passed.
Design and analysis peer reviews: `extension_design_review.md` and
`extension_analysis_audit.md`.
