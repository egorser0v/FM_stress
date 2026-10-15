# Kernel and window-length scan: measured findings

Status: completed on 2026-10-12. All 360 fits ran on CUDA (RTX 2060), and none was excluded. The protocol is `docs/kernel_scan_protocol.md`. It was amended before any scored run in two ways: the device became CUDA, and the scan ran as three parallel processes, one per data draw.

Verification:

- **Independent audit** (`scripts/check_kernel_scan.py`): 360/360 runs, 14,682 checks, no failure. CPU replay of the saved checkpoints reproduces the saved predictions to a relative RMS of at most 2.6·10⁻⁶.
- **Analysis verification** (`scripts/check_kernel_scan_analysis.py`): 199 checks, no failure.

This study is supplementary and exploratory. It does not change any earlier result, pass flag or conclusion, and its CUDA runs are never pooled with the earlier MPS studies.

## 1. Pre-declared verdicts (§5.3, GP source, S4 vs MLP)

| Endpoint | Verdict | Why |
|---|---|---|
| E1, residual-PC velocity MSE | **Partial** | (a) and (b) hold, (c) fails. Details below. |
| E2, fair CRPS | **Refuted** | No consistent trend, no kernel contrast, no S4 advantage in any cell. All S4/MLP ratios lie between 0.978 and 1.004. |

E1 in detail:

- **(a) holds.** The horizon-axis trend favours S4 for all four kernels.
- **(b) holds.** The OU − RBF contrast favours S4 in every draw (mean log ratio −0.04).
- **(c) fails.** No cell with F/ℓ ≥ 4 shows an S4 advantage. The lowest ratio is 0.989 (OU, F = 64).

**What actually moves under E1** is S4's disadvantage at the RBF anchor (S4/MLP = 1.18) disappearing. An S4 advantage never appears. On the axis attribution, RBF and both Matérn kernels show the trend on both axes, which makes it an F/ℓ effect. OU shows it on the horizon axis only.

**Recommendation for the latent Gauss–Markov programme: keep TimeFlow's head.** In none of the six Gauss–Markov cells (Matérn-5/2, Matérn-3/2 and OU at F = 32 and 64) does S4 or the whitened MLP have a material advantage over the AdaLN MLP, on either endpoint.

## 2. The assignment's labels

- **Literal Task 2: "Empty task" in all 20 kernel × geometry cells.** No run among the 360 passes the literal 20% rule.
- **Head comparison in the three cells where the assignment's reading applies** (cells with the numerical contrast: RBF, Matérn-5/2 and Matérn-3/2 at F/ℓ = 2): **"Prior was overstated"** in all three. The MLP matches or beats S4 on PC1 error, residual error and roughness distance.
- **Head comparison in the other cells.** Most are also "Prior was overstated". Three cells are labelled "Head, not prior": RBF at F16_l2, Matérn-5/2 at F16_l4 and OU at F64_l8. That label comes **only** from roughness distance: S4 is closer to the target roughness, consistently across draws. Both heads are within 1–4% of the target, far inside the ±20% band, so a ratio of such small distances is a fragile criterion. Using it with the 0.8 velocity threshold is the analyzer's documented reading of §5.4. A 0.95 threshold would give the same labels.
- **Task 3.** The whitened MLP has an advantage only at the RBF anchor, on E1: whitened/MLP = 0.79 and whitened/S4 = 0.68, with CRPS a match. Everywhere else the label is "Third head and S4 tie raw MLP".
- **Anchor replication.** At RBF F16_l8, S4/MLP is 1.18 on residual MSE (main study: 0.05827 / 0.04927 = 1.18) and 0.978 on CRPS (mixture study: 0.97). The anchor reproduces the earlier studies.

## 3. Why the scan resolves so little

In rough kernels and long windows both heads come close to the privileged oracle. Most of their error is irreducible conditional noise from independent coupling, and ratios of total error are compressed towards 1.

Mean model/oracle ratio, GP source:

| Cell | Residual, MLP | Residual, S4 | CRPS, MLP | CRPS, S4 |
|---|---|---|---|---|
| RBF F16_l8 (anchor) | 1.58 | 1.86 | 1.41 | 1.38 |
| Matérn-3/2 F64_l8 | 1.03 | 1.03 | 1.06 | 1.06 |
| OU F64_l8 | 1.03 | 1.02 | 1.05 | 1.05 |

The reducible part is largest exactly at the near-rank-2 RBF anchor. That is the only place where the earlier studies found head differences, and the only place where whitening helps.

## 4. Exploratory, post hoc: excess over the oracle

This analysis is not a protocol endpoint. The protocol lists oracle gaps as secondary, but it defines no rule for comparing them between heads. The quantity is S4 excess / MLP excess over the oracle, per draw, GP source.

| Kernel | F16_l8 (F/ℓ = 2) | F64_l8 (F/ℓ = 8) | F16_l2 (F/ℓ = 8) |
|---|---|---|---|
| RBF | 1.41 / 1.56 / 1.46 | 1.04 / 1.03 / 0.99 | 1.03 / 1.05 / 0.99 |
| Matérn-5/2 | 1.04 / 1.31 / 1.20 | **0.87 / 0.89 / 0.95** | 0.92 / 1.03 / 0.94 |
| Matérn-3/2 | 1.00 / 1.16 / 1.00 | **0.84 / 0.86 / 0.84** | 0.86 / 1.02 / 0.89 |
| OU | 1.15 / 1.02 / 0.86 | **0.55 / 0.62 / 0.54** | 0.88 / 1.06 / 0.79 |

- **Where S4 leads.** At the longest window (F = 64), S4 removes more of the reducible residual error than the MLP for all three Gauss–Markov kernels, in every draw.
- **Not an F/ℓ effect.** At the same F/ℓ on the lengthscale axis (F = 16), the means are lower but one draw in three goes the other way. The window length matters, not F/ℓ.
- **Small in absolute terms.** The difference is at most about 1.2% of the total residual error.
- **No forecast counterpart.** The CRPS excess ratios at F = 64 are 0.93–0.96 and are mixed elsewhere.

This is a lead for a pre-declared follow-up with longer windows (F ≥ 64–128) and longer training. It is not a finding.

## 5. White source, outside the verdicts

At RBF F64_l8 with the white source, S4 has a material residual-error advantage over the MLP (0.70), while the whitened MLP is worse than the MLP (1.49). No other white-source cell shows an S4 advantage on E1 or E2.

## 6. Limits

- **Three data draws per cell.** Kernels share common random numbers, so kernel contrasts are not independent replications.
- **Fixed budget.** About half of the selected checkpoints sit at the last update, between 8 and 16 of 24 per head × geometry. S4 at F = 32 and 64 has the fewest (8–9 of 24), so longer windows may reward more training.
- **CUDA numerics** differ from the earlier MPS studies. The anchor agrees, but the runs are not pooled.
- **Narrow source.** The GP source is narrower than the normalized target (main's convention; addendum §8).
- **Fragile secondary label.** "Head, not prior" rests on roughness distances of a few percent (§2).

## 7. What this means for the project question

Across four kernels, from super-smooth RBF to OU, and window-to-lengthscale ratios from 2 to 8, a tiny AdaLN MLP and a matched three-block S4 reach the same forecast quality at this budget.

The main study's S4 disadvantage on residual PCs is specific to the near-rank-2 RBF anchor and vanishes elsewhere. So does whitening's benefit. Nothing in the scan supports switching to an S4 head for Gauss–Markov priors, and the pre-declared recommendation is to keep TimeFlow's head. The one open lead is the post-hoc reducible-error advantage of S4 at long windows (§4).

## Artifacts and reproduction

- Results: `results/kernel-scan/` (about 1 GB) and the run logs `results/kernel-scan-training-*.log`.
- Audit: `results/kernel-scan/independent_verification.json` and `results/diagnostics/kernel_scan_audit.log`.
- Analysis: `output/kernel-scan/` contains:
  - `summary.json`;
  - `cell_contrasts.csv` (every metric, source and head pair);
  - `runs.csv`;
  - `exploratory_oracle_excess.csv`;
  - two figures.
- Analysis check: `results/diagnostics/kernel_scan_analysis_verification.json`.

```sh
.venv/bin/python -m fm_stress.kernel_scan --device cuda --data-seed 307041   # one process per data draw
.venv/bin/python scripts/check_kernel_scan.py
MPLCONFIGDIR=/tmp/fmstress-mpl .venv/bin/python -m fm_stress.analyze_kernel_scan
python3 scripts/check_kernel_scan_analysis.py
```
