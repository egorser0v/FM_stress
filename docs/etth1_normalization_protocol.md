# E4: global versus history normalization on ETTh1-OT — protocol

Status: written on 2026-10-13, before any training for this experiment. This is a supplementary, exploratory study, experiment E4 of `docs/etth1_level_hypothesis.tex`. The configuration is `configs/etth1_normalization.json`; the runner is `fm_stress/etth1_normalization.py`.

## Question

Under global normalization, the prior × head interaction on ETTh1-OT is large. MLP gets worse with the GP prior, and S4 gets better.

The probes traced this to the history → forecast-level mapping on the cold test period, where it stops following the history:

- MLP with the white prior follows with a gain of 0.62;
- MLP with the GP prior, 0.15;
- S4 with the GP prior, 0.35.

History (Sundial) normalization makes level-following exact by construction, so the forecast moves one-to-one with the history level. **Does the prior × head interaction disappear when the absolute level is no longer a network input (condition C2)?**

## Design

24 fits on one device: CUDA, RTX 2060, host `home1`.

- **Normalization** ∈ {global, history}:
  - global: train-scaled windows, as in the extension and E5;
  - history: each window divided by its own history mean and standard deviation, with the threshold 0.01 → 1 as in `data.sundial_normalize`.
- **Prior** ∈ {white, GP}: drawn in the model's coordinates, with the same covariances as the extension.
- **Head** ∈ {MLP, S4}: width 55, as in the extension.
- **Seed** ∈ {31, 32, 33}.

Training, the RNG order, checkpoint selection and evaluation draws are identical to E5.

**Scoring.** History-normalized forecasts are mapped back to train-standardized units as μ_h + s_h · x. Every forecast is scored with fair CRPS on the same 64 test contexts, against the same targets.

## Reading rules (fixed now)

The interaction for a normalization is

  I(norm) = [CRPS(S4, gp) − CRPS(MLP, gp)] − [CRPS(S4, white) − CRPS(MLP, white)],

using means over seeds.

1. **Reference.** The global cells must reproduce the known pattern: MLP-gp ≥ 1.10 × MLP-white and S4-gp < MLP-gp. Otherwise E4 is uninterpretable.
2. **C2 supported** if |I(history)| ≤ 0.5 · |I(global)| **and**, under history normalization, MLP-gp is within ±5% of MLP-white.
3. **C2 refuted** if |I(history)| ≥ 0.8 · |I(global)| with the same sign.
4. **Otherwise inconclusive.**

## Secondary reporting

- Absolute CRPS per cell under both normalizations, against the ridge-with-residuals baseline (0.093).
- 90% coverage and width.
- A check that history-normalized forecasts follow a history shift exactly.

## Limits

- One series, one test period and 64 contexts.
- The three seeds share the data.
- The budget is fixed at 2,000 updates.
- CUDA numerics differ from the earlier MPS runs, so E4's own global cells serve as its reference.
