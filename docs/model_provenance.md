# Model provenance and exact experimental adaptations

## What is implemented

`fm_stress/models.py` contains two **small velocity predictors trained from
scratch**, not pretrained Sundial or a full reproduction of TSFlow. Both accept
`forward(x, t, history)`, with shapes `[batch, horizon]`, `[batch]` (or `[batch,1]`),
and `[batch, lookback]`, and return a velocity of shape `[batch,horizon]`.

The four main cells cross independent white/GP sources with AdaLN MLP/S4 heads.
Whitening is a coordinate transform around the same MLP architecture; it is not
an additional architecture in this file. The runner must transform input,
velocity targets and predictions consistently and evaluate in original
observation coordinates.

### AdaLN MLP: official TimeFlow architecture

- Official [Sundial repository](https://github.com/thuml/Sundial), Apache-2.0.
- Actual released head: [HF `flow_loss.py`](https://huggingface.co/thuml/sundial-base-128m/blob/3212e42564493f520593e5414af4367fc4b49226/flow_loss.py),
  model revision `3212e42564493f520593e5414af4367fc4b49226`.
- Unmodified reference and license are preserved in `vendor/sundial/`.
- Adapted components: sinusoidal time embedding, global input projection,
  adaptive-LayerNorm residual MLP blocks with learned shift/scale/gate,
  conditional final LayerNorm, and zero-initialized output/modulation.
- Changes for this experiment: a small lookback encoder replaces the
  pretrained Transformer; hidden widths/depth are configurable; Fourier
  embedding width is reduced from 256 to 32; prediction dimension is the
  entire future patch. Time is supplied as `t in [0,1]` and multiplied by 1000
  inside the shared embedding, matching TimeFlow's head time convention.

**Important objective difference.** The released HF `FlowLoss` predicts the
target `y` with a horizon-weighted squared loss, and samples with
`x += (prediction - initial_noise) * dt`. The project explicitly requests
velocity `u = y - epsilon` and squared velocity error. Therefore only the
**architecture** is adapted: the experiment runner uses the project's velocity
target and objective, not the released `FlowLoss` training/sampling functions.
Results must not be described as failures of pretrained Sundial or as a direct
Sundial benchmark reproduction.

### Temporal head: genuine S4-NPLR, TSFlow-style residual backbone

- Official [state-spaces/s4](https://github.com/state-spaces/s4) revision
  `e757cef57d89e448c413de7325ed5601aceaac13`, Apache-2.0.
- [Standalone S4 source](https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/models/s4/s4.py)
  and license are preserved unmodified in `vendor/state_spaces_s4/`.
- Adapted official mathematical construction: HiPPO-LegS initialization,
  rank-one normal-plus-low-rank correction, conjugate eigenbasis,
  negative-real-part parameterization, learned step size, and bilinear
  discretization. Initialization does **not** apply the optional `B_clip=2`
  used by the current state-spaces helper; it preserves the exact HiPPO input
  vector, consistent with the older helper copied into TSFlow.
- Official [TSFlow backbone](https://github.com/marcelkollovieh/TSFlow/blob/ff893176a41a1037f63efae674aab71480eee7d0/tsflow/arch/backbones.py)
  and [its S4 implementation](https://github.com/marcelkollovieh/TSFlow/blob/ff893176a41a1037f63efae674aab71480eee7d0/tsflow/arch/s4.py)
  were inspected at revision `ff893176a41a1037f63efae674aab71480eee7d0`.
  TSFlow source itself is not vendored because its repository tree does not
  provide a license file. The surrounding small residual backbone is an
  independent implementation of the published design.
- Preserved design: **three** temporal residual blocks, bidirectional S4,
  pre-LayerNorm temporal mixing, GELU and pointwise output mixing, time
  conditioning, `tanh(z) * sigmoid(z)` gating, residual and skip branches,
  sum of skip branches, final pointwise MLP.
- Explicit reductions/adaptations: univariate only; no feature-axis attention;
  shared small lookback encoder rather than TSFlow's original feature pipeline;
  no dropout; state dimension defaults to **16 real states**, rather than
  TSFlow's 128, to suit this deliberately small horizon-16 comparison; last
  unused residual projection is omitted; final velocity projection starts at
  zero so both families initially predict zero velocity. No subtract-input
  (`init_skip`) prediction convention is used because output is directly a
  velocity. Both families use the same Fourier time embedding implementation.

This is **S4 with a learned rank-one correction**, not S4D and not a generic
learned convolution. It is a reduced TSFlow-style backbone, not all of TSFlow.

## Why the S4 implementation runs on Apple MPS

The official efficient implementation uses complex Cauchy kernels and FFTs.
Those are unnecessary at this experiment's short horizon. We evaluate the
same finite SSM kernel through real state matrices instead.

For a retained half of the conjugate states, write `z = r + i s`,
`w = a + i omega` and `p = p_r + i p_i`. The real continuous state matrix is

```
A_real = [[diag(a), -diag(omega)],
          [diag(omega), diag(a)]] - 2 [p_r; p_i] [p_r; p_i]^T.
```

The factor 2 accounts for the conjugate half. With `h = dt/2`, bilinear
discretization gives `A_bar = 2 (I-h A_real)^(-1) - I` and
`B_bar = (I-h A_real)^(-1) dt B_real`. The inverse is evaluated using real
2-by-2 block inverses and a rank-one Sherman-Morrison update. It is exact up
to floating-point roundoff and does not use a numerical ODE approximation
for the internal S4 kernel.

The finite kernel `K[k] = C_real A_bar^k B_bar` is constructed by a doubling
Krylov calculation and applied as a short dense temporal convolution. The
backward half uses precisely TSFlow's FFT padding/index convention (including
its one-position offset). The resulting layer remains noncausal in the
future patch, as required for bidirectional S4.

CPU complex operations are used **once at construction** to diagonalize the
HiPPO matrix. All trainable forward/backward operations use real tensors on
the chosen device. Training does not copy parameters to CPU or silently fall
back to a CPU kernel. Complexity differs from optimized official S4: this
implementation targets short patches and is not appropriate evidence for
long-sequence speed or scaling claims.

## Fair comparison and encoder interpretation

At horizon 16, lookback 32, encoder width 32, model width 56, three blocks,
and state size 16, trainable counts are:

| Model | Parameters |
|---|---:|
| AdaLN MLP | 65,464 |
| S4 temporal head | 65,281 |

The difference is 0.28%. Counts include encoder and time embedding and exclude
dead parameters. Recompute with `count_parameters` if configuration changes.
Both encoders use `Linear(lookback,32) -> SiLU -> Linear(32,32)`. The runner
should copy encoder initial weights for all cells at each seed, then jointly
train each model separately. "Same encoder" thus means **same architecture
and initial weights**, not one frozen encoder with identical learned weights.
Encoder adaptation can interact with the head; report this limitation rather
than attributing every difference exclusively to the head's final layer.

Use the same optimizer budget and report parameter counts and elapsed times.
`optimizer_groups` preserves the official no-weight-decay treatment of SSM
transition/input/step-size parameters. No forced smaller learning rate is
imposed on S4; any chosen rate must be part of the recorded protocol.

## Numerical verification

`tests/test_models.py` checks:

1. Real finite-horizon S4 kernels against an independently implemented full
   complex bilinear SSM, in float64, including gradients of every parameter.
2. HiPPO initialization invariants.
3. Dense bidirectional temporal mixing against the official FFT/padding
   convention, in float64.
4. Both model interfaces, t=0/t=1, finite gradients, and actual parameter updates.
5. CPU/MPS kernel and backward agreement when MPS is exposed to the process.

On the current Apple M3 Pro machine, all six tests passed on 2026-10-04 using
PyTorch 2.14.1. A sandbox without Metal access reports MPS unavailable and skips
the sixth test; this is not proof that the machine lacks an MPS device.

An indicative 20-step warmed-up benchmark with batch 128, width 56, 16-state
S4, float32 and Adam measured approximately 4.1 ms/MLP step and 11.6 ms/S4 step
on MPS (CPU, four threads: 1.6 and 12.3 ms). These are local sizing measurements,
not research benchmark results. Exact wall times vary with load and batch size.
