# Independent design review: sparse, convolutional and multivariate extensions

Reviewed 8 October 2026, before extension outcomes. The original synthetic
univariate experiments and their operational success flags remain immutable.
This note distinguishes the proposed extension from direct paper reproduction.

## Connection to primary sources

Sundial's paper describes univariate pretraining, a Transformer history
representation and a small TimeFlow MLP. A jointly multivariate tiny head is
therefore our adaptation, rather than a direct evaluation of pretrained Sundial.
[Sundial, §4](https://arxiv.org/html/2502.00816v2).

TSFlow's revised paper includes a multivariate extension with a Transformer
across the feature axis and compares it with channel-independent modeling.
It evaluates multivariate forecasts with CRPS-Sum. Its main training recipe also
uses substantially more updates, EMA, conditional priors and dataset-specific
conditioning than this small independent-coupling study. A pointwise channel mixer
plus temporal S4 is related to that design but must not be described as its exact
multivariate implementation. [TSFlow, Appendix B.8 and A.2](https://arxiv.org/html/2410.03024v2#A2.SS8).

ETTh1 supplies hourly oil temperature and six load variables. These make an
oil-temperature-only task and a genuine seven-channel forecasting task from the
same real source. They are related evaluations, not two independent real datasets.
[Official ETDataset repository](https://github.com/zhouhaoyi/ETDataset).
Exchange supplies eight daily currency series and is also a TSFlow evaluation
family; its L=32/F=16 protocol here need not match the paper's benchmark.
[Original multivariate data repository](https://github.com/laiguokun/multivariate-time-series-data).

The official Time-Series-Library ETT loader uses fixed chronological boundaries
at 12, 16 and 20 blocks of 30 days and fits its scaler on training data. A new
strictly disjoint-window variant should disclose deviations instead of claiming
identical benchmark scores. [Official loader](https://github.com/thuml/Time-Series-Library/blob/main/data_provider/data_loader.py).

## Reviewed bounded suite

The parent agent proposes four tasks, four heads, two sources and three seeds:

| Task | History | Horizon | Channels |
|---|---:|---:|---:|
| Correlated synthetic GP, cross-channel correlation 0.6 | 32 | 16 | 3 |
| ETTh1 oil temperature | 48 | 24 | 1 |
| ETTh1 all measured variables | 48 | 24 | 7 |
| Exchange | 32 | 16 | 8 |

Heads are dense AdaLN MLP, fixed-mask sparse MLP, three-block S4 and a small
one-dimensional U-Net. The 96 planned fits use 2,000 updates, LR 0.001, a
validation-selected checkpoint and approximately matched active parameter
counts. This is a fixed-budget comparison, not a convergence or optimal-tuning
claim. Prior learning-rate selection on another experiment is motivation, not
fresh validation evidence that this rate is optimal for U-Net or sparse MLP.
Benchmark per-step and sample-generation runtime before launch; reduce any
declared budget only before scored training and document it. Record actual MPS
runtime, dense storage size and active parameter counts. Do not claim a 20–40
minute completion time until measured throughput supports it.

## Required fairness and leakage checks

**Sparsity must be explicit.** A fixed binary weight mask is a weight-sparse
network, not a mixture of experts or sparse activations. State which layers are
masked, density, mask seed and whether masks ever change. Check that masked
weights contribute zero and receive zero effective gradients. Multiplying dense
weights by a mask generally retains dense computation; do not claim an actual
speedup. Matching active parameter counts may require a wider sparse hidden
layer, so this compares matched-active-budget architectures rather than isolating
sparsity at identical width. Report both interpretations.

**Joint means joint.** Every multivariate model must consume all historical
channels and output the full F×C future jointly. S4 operates along time and
mixes channels; it must not convolve along a flattened interleaving of channel
and time indices. A U-Net can use noncausal convolutions over the noisy future
patch without leaking the true future: interpolant coordinates are inputs to the
flow field. The separately supplied conditioning must contain history only.

**Hold source channel structure fixed.** The proposed pair is temporally white
versus temporal RBF-GP, both with the same training-estimated channel correlation.
In multiple channels the former is not isotropic white noise. Keep diagonal
variance equal, channel-covariance regularization explicit and all fits train-only.
Use a documented flattening convention and test the Kronecker/separable sampler
against empirical time and channel covariances. On real data a fixed RBF prior is
a smooth prior, not a demonstrated matched target distribution.

**Split before constructing windows.** Use chronological real-data splits and
disjoint future targets across train/validation/test. A stricter fully disjoint
history-plus-target split is acceptable but should be identified. Fit global
channel scales, PCA, prior correlation, ridge coefficients, residual pools and
all data-dependent choices on training data only; choose ridge regularization
on validation only. Instance statistics use each available history, never its
future. Persist forecast-origin indices and exact dataset hashes.

**Report scores in fixed units.** The primary real-data proper scores should use
inverse-history-normalized predictions/targets in train-global-standardized
channel units. This prevents changes in per-window history scale from silently
changing the weighting of forecast difficulty, and prevents high-unit channels
from dominating joint scores. Also expose per-channel results. If normalized
instance-space scores are supplied, label their different weighting explicitly.

## Metrics, baselines and uncertainty

Use fair ensemble marginal CRPS plus fair energy score on the flattened **joint
F×C path**. Computing energy separately per channel and averaging would not test
cross-channel dependence. An optional variogram or cross-channel covariance
diagnostic can complement energy score, which may be insensitive to some
dependence errors. The planned 32 samples and Euler-64 are reasonable bounded
choices; Euler-128 checks on the same 16 fixed contexts/source draws test a
specific solver sensitivity, not complete convergence.

Use persistence and the declared seasonal-repeat baseline where meaningful,
plus a train-fitted ridge predictor selected by validation. A ridge-plus-training-
residual bootstrap gives a cheap probabilistic baseline. Draw complete residual
F×C vectors so temporal and cross-channel residual structure are preserved;
independent coordinate noise would define a different baseline. Training residuals
can underestimate predictive uncertainty, so label this simple baseline rather
than treating it as a calibrated oracle. A five-step repeat on Exchange is a
declared lag baseline, not evidence that the series has a five-day seasonal law.

Velocity-PC metrics remain useful within a source. Do not infer source
superiority by comparing their raw values across sources: source changes both
the label distribution and its PCA. Compare source effects and head-by-source
interactions using proper scores against the same target draw. Avoid pooling
scores across datasets/channels with different scales and dimensions.

For real ordered histories, an iid history bootstrap is unjustified even when
target windows do not overlap: serial dependence can remain. Use a declared
moving-block bootstrap, or restrict uncertainty to the three paired optimization
seeds conditional on this fixed test period. Neither gives uncertainty across
training periods, hyperparameter searches or datasets. Report seed-level values
and avoid definitive superiority/equivalence claims from three runs.

## Geometry and conclusions

Keep target and both velocity spectra train-fitted and test-reported. In a
multivariate smooth process, approximately two temporal components **per channel**
can imply rank near 2C globally; global E2 need not exceed 0.9. Report E2, E(2C)
and per-channel E2 without manufacturing almost-identical channels to pass the
old univariate gate. Real data that fail the old near-collinearity gate remain
valid broader-extension tasks at the user's request, but do not count as a
confirmatory replication of the original smooth-geometry claim.

The useful outcome is a qualified source/head comparison across these small
settings, including null findings and classical-baseline wins. It is not a claim
that full Sundial or TSFlow has been reproduced or that a new architecture is
universally preferable. No implementation or result audit is implied by this
design review; the new runner, tests and completed artifacts require inspection.

## Initial implementation review

Read `extended_data.py`, `extended_metrics.py` and `extended_models.py` as they
were prepared. No blocking issue found in the reviewed version. The separable
synthetic sampler and Kronecker source covariance use the same time-major,
channels-last ordering. Real windows stay fully inside chronological splits;
the scaler uses unique training rows, and source correlation uses training
targets only. Joint energy flattens F×C, while roughness differences are taken
along time rather than across channel boundaries. Energy is divided by
sqrt(F×C), and channel-sum CRPS by sqrt(C); these are disclosed fixed rescalings,
not numerically interchangeable with published unscaled scores.

The implemented extension currently uses **global training standardization
only**, with no per-history instance normalization. This makes scores comparable
across windows but is a departure from the original Sundial-normalized stress
geometry and must remain explicit. None of the preceding conditional advice
about inverse instance normalization implies that such a transform is present.

Models consume complete joint histories through an identity representation.
The static sparse MLP masks all linear maps in its core, including conditioning
and time maps, with exact per-output row connectivity; it does not only sparsify
the residual hidden layers. U-Net genuinely downsamples/upsamples time and fuses
skip features; S4 applies its audited state-space operation on time with joint
channel projections. A smaller channel mixer is not TSFlow's feature Transformer.
GPU tests, runner RNG pairing, checkpoint selection, scores and final artifacts
remain to be verified once their implementation is ready.

## Runner prelaunch audit

Read the full `extended_experiment.py` and `configs/extension.json`. The finalized
evaluation selects 64 evenly spaced test contexts, 32 source draws per context,
Euler-64, and Euler-128 on the first **eight** selected contexts for the first
training seed. This supersedes the earlier suggested 16-context sensitivity.
Training seeds are 31, 32 and 33; static mask density is 0.5. Width is selected
using active parameter count only, with no outcome-driven capacity tuning.

No blocking leakage, RNG, axis or score-computation error was found. Training
randomness is reset after architecture construction; PCA and ridge fitting are
training-only; ridge alpha and neural checkpoints use validation; baseline and
neural scores use the identical selected targets. Whole-vector residual draws
preserve the fitted baseline's residual dependence. Source/configuration/data
hashes protect result reuse. The runner uses fixed global training units and
correctly keeps joint sampling and temporal differences distinct.

Independent CPU execution passed all 24 extended data/model/metric tests. A
separate one-update CPU smoke run for all four heads completed checkpointing,
evaluation, generation and solver comparison. Forward hooks verified that the
first training interpolant, flow time and history were bit-identical across all
four heads after their differing initialization. These temporary smoke outputs
are not research results and are not evidence of MPS execution. No runner or
numerical module was changed by this reviewer.

One inexpensive prelaunch omission was reported to the parent: the inspected
runner recorded both velocity spectra but not the target spectrum. Adding a
train-fitted target PCA with validation E2/E(2C)/per-channel energy preserves the
connection to the original geometry audit. Final result inspection must also
retain the limited sample count, short training budget and fixed-test-period
interpretation of seed uncertainty.

## Final analysis and figure review (8 October 2026)

The strict analyzer completed on all 96 declared fits. It checked the actual
frozen numerical-source and real-data hashes, run identity/budget/completeness,
validation-minimum selection, archive shapes, fixed selected test indices and
stored per-context score means. `output/extension-analysis/summary.json` is
complete: 32 cell aggregates, 104 paired-seed contrasts and 182 context-bootstrap
rows. Raw-array/checkpoint recomputation is an independent verifier's separate
responsibility; this aggregation review does not claim to replace it.

Source-by-head contrasts use `(head GP - MLP GP) - (head white - MLP white)`
for each proper score, pairing all four cells before bootstrap. Hand-calculated
constant-difference tests verified signs and pairing. ETTh1 oil-temperature CRPS
has an S4 crossover: white S4 0.170448 versus MLP 0.142222; GP S4 0.143717 versus
MLP 0.194988. The interaction is -0.079497; fixed-model averaged-context intervals
are [-0.143591, -0.028409] with circular blocks of four selected origins and
[-0.149368, -0.022512] with blocks of eight. These exploratory intervals omit
training-dataset and model-selection uncertainty. They are not population-level
proof, especially with many unadjusted comparisons.

The corresponding seven-channel ETTh1 interaction is block-sensitive: its
block-four interval excludes zero narrowly, whereas block-eight includes zero.
Synthetic GP and Exchange S4 interaction intervals also include zero. Thus the
univariate crossover does not establish a universal benefit of S4 or GP sources.
Sparse MLP has the best neural mean CRPS on Exchange, but active-capacity matching
changes its width and does not isolate sparsity at fixed architecture. Every
neural cell loses to the ridge-plus-training-residual baseline on both proper-score
means in all four datasets; the limited 2,000-update budget and optimization
choices remain material limitations.

The OT-only comparison checks exact equality of scalers, targets and forecast
origins before scoring the multivariate model's OT output separately. Seven of
eight joint-output configurations have worse mean OT CRPS than the corresponding
univariate configuration; U-Net with white source has a small reverse difference
whose seed interval crosses zero. This comparison changes conditioning, output
dimensionality, width/capacity allocation and source covariance, so it is not a
single-factor causal test of access to additional channels.

All eight standalone PNG figures (two proper-score panels, geometry, learning
curves and four fixed-context path grids) were visually inspected. Axis labels,
legends and crops are clear; long ETTh1 path labels were checked additionally at
native resolution. Matching vector PDFs were saved. Score error bars are seed SD;
path bands are marginal 90% intervals for the first three declared evaluation
contexts and first seed, with a shared scale within each row and no outcome-based
selection. Analysis and figure review were CPU-only and changed no frozen
numerical source or model output.
