# Sparse MLP, U-Net, multivariate and real-series extension

Prospective configuration: `configs/extension.json`. This is a separate,
exploratory extension, not a replacement for Tasks 1–3 or a replication of
Sundial/TSFlow benchmark scores. Original code, checkpoints and results remain
unchanged. All comparisons here use fresh runs of all four heads.

## Questions and design

Does the source–head interaction persist with sparse connections, temporal
convolutions, multiple jointly predicted variables, and real targets?

Four datasets × four heads × two sources × three seeds = 96 fits. Every fit
receives 2,000 updates of 128 windows, AdamW with learning rate 0.001, weight decay
0.0001, gradient norm limit 1. Validation velocity MSE is checked every 250 steps;
the lowest-validation checkpoint is evaluated. There is no learning-rate search
in this extension. The rate follows the preceding project pilots, but this is
not proof that it is equally optimal for new heads. Fixed budget does not imply
convergence. Seed values are 31, 32, 33.

All heads see the full flattened history through an identity encoder. Dense and
sparse MLP jointly mix horizon/channel coordinates. S4 retains the validated
rank-one HiPPO/NPLR temporal operator with channel-mixing projections. The new
1-D U-Net has three temporal resolution levels, two down/up stages and skip
connections; it is an independent compact implementation, not copied from either
paper. Sparse MLP masks all linear maps, including conditioning/time/output
maps, with approximately 50% active connections and fixed row-wise masks. It is
not a mixture of experts and does not use sparse compute kernels. Masks vary
with the model seed and are saved in checkpoints.

Widths minimize distance to 64,000 **active** scalar parameters, using only
architecture counts and integer widths 4–112. Counts and allocated parameters
are saved. This is approximate capacity matching, not equal FLOPs, memory or
runtime. Sparse MLP allocates roughly twice its active weights. Report runtime
as well as counts; do not claim acceleration from masked dense operations.

## Data and provenance

| Dataset | History → future | Channels | Split/window protocol |
|---|---:|---:|---|
| Correlated synthetic GP | 32 → 16 | 3 | 8,192/1,024/1,024 independently generated windows; RBF lengthscale 8, channel equicorrelation 0.6 |
| ETTh1 oil temperature | 48 → 24 | 1 (OT) | 12/4/4 nominal 30-day months; train/val/test strides 4/24/24 |
| ETTh1 joint | 48 → 24 | All 7 | Exactly the same timestamps/splits as OT experiment |
| Exchange joint | 32 → 16 | All 8 | First 70%/next 10%/last 20%; strides 1/16/16 |

ETTh1 is a dataset evaluated in [Sundial, Tables 1 and 7](https://arxiv.org/html/2502.00816v2).
Exchange appears in [TSFlow, including Appendix B.8 multivariate evaluation](https://arxiv.org/html/2410.03024v2).
Sundial uses univariate pretraining and a large pretrained Transformer; TSFlow's
multivariate extension uses feature attention. Neither full architecture is
replicated here. Our short horizons, small trained-from-scratch heads, chosen
splits and metric scales differ from the papers.

Download pinned author repositories using `scripts/fetch_extension_data.py`.
Exact URLs, revisions and SHA-256 hashes are in `data/real/manifest.json` and the
experiment manifest. ETTh1: [ETDataset](https://github.com/zhouhaoyi/ETDataset),
Haoyi Zhou et al., *Informer*, AAAI 2021; repository license CC BY-ND 4.0.
Exchange: [Lai et al.'s multivariate datasets](https://github.com/laiguokun/multivariate-time-series-data),
*Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks*.
Raw data are cached locally, excluded from Git, and reproducibly downloaded;
no redistribution license is assumed for the Exchange repository.

Real series are split before windowing. Entire history+future windows must lie
within a split, so no training target or held-out history crosses a boundary.
ETTh1 uses row ends 8640/11520/14400; the remaining dataset tail is unused.
Actual split sizes: ETTh1 2143/118/118 windows; Exchange 5264/45/92.
Test targets do not overlap, but temporal dependence may remain.

Per-channel mean/standard deviation are fitted to the raw training segment only
(all training-window points for independent synthetic draws). There is **no
per-window history normalization** in this extension. Scores are in these fixed
train-standardized units, so units are comparable within a dataset. This differs
from the original project's Sundial-style instance normalization; new and old
numerical tables must not be pooled.

## Source and objective

Independent source/target coupling with straight paths:
`x_t = (1-t) epsilon + t y`, target velocity `y-epsilon`, raw mean squared velocity
loss. Both sources have unit marginal channel variance and the same train-target
channel correlation, regularized 5% towards identity.

* Temporally white: `I_F ⊗ C_train` (not fully isotropic when channels correlate).
* Smooth GP: normalized nugget-regularized RBF(lengthscale 8) `⊗ C_train`.

This isolates temporal source covariance while holding cross-channel covariance
fixed. Source lengthscale is fixed prospectively, not optimized for either real
dataset. A GP source on real targets is a modeling assumption, not a claim that
real targets follow a GP. Training minibatches, standard-normal source draws and
times are paired across heads after model initialization. Source-specific
training PCA is fitted in joint, time-major `[F,C]` coordinates.

Target and velocity validation spectra report top-2, top-2C and per-channel
top-2 energy. The original univariate rank-two gate is **not imposed** on real or
multivariate data, and no channels are made nearly identical to force it to pass.
These are generalization experiments, not additional rank-two stress-test passes.

## Held-out evaluation

Velocity PC1/PC2/residual (PC3 onward) errors retain the original definitions,
with joint F×C PCA axes and raw versus variance-normalized values distinguished.
Velocity labels change with the source; source preference is assessed primarily
with forecast scores on the same targets, not by cross-source velocity MSE.

Use 64 test contexts selected evenly across each test split, 32 independent
conditional generations per context, 64 Euler steps. All models use paired
contexts, random sources and evaluation times. On seed 31, the first eight
selected contexts are also evaluated at 128 Euler steps without replacing the
primary scores. This is a limited solver diagnostic, not a convergence proof.

Report marginal fair CRPS, joint energy score divided by sqrt(F×C), channel-sum
fair CRPS divided by sqrt(C), ensemble-mean MSE, marginal 90% interval coverage
and width, and temporal roughness. Off-diagonal U-statistics correct finite
ensemble CRPS/energy bias. Roughness differences are **along time only**, never
between adjacent flattened channels. Scores do not share the normalization of
published Sundial/TSFlow tables.

Baseline forecasts: persistence, daily seasonal naive for hourly ETTh1, ridge
regression (alpha from 0.1/1/10/100 by validation MSE), and ridge plus whole-vector
training-residual resampling. The latter preserves residual cross-channel/time
dependence but uses in-sample residuals and may understate uncertainty. All point
baselines are reported as degenerate predictive distributions when scored with
proper scores; zero interval width must not be mistaken for calibrated forecasts.

Summaries show all three seeds and their SD, conditional on a fixed test subset.
Do not treat ensemble members, channels, horizon steps, or real test windows as
independent experiments. Optional exploratory paired context intervals use
ordered moving blocks of length 4, with length 8 sensitivity for real data;
synthetic contexts may use IID resampling. Three seeds and two real datasets
cannot establish universal architecture superiority. The ETTh1 multivariate
model can additionally be scored on OT alone using exactly the OT-only test
origins and train scaler; all-channel averages are not comparable to OT-only MSE.
