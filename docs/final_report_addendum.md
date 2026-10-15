# Post-hoc addendum to the final research report

Status: post-hoc review written on 2026-10-12, after every study in this repository was complete. It does not change the declared primary flags, `docs/conclusions.md`, the report PDFs or any saved result. It adds interpretation that the final report (`output/pdf/research-report.pdf`) lacks.

Each number below is recomputed from saved arrays by `scripts/report_addendum_numbers.py`. That script uses NumPy only and imports no `fm_stress` code. It writes `results/diagnostics/report_addendum.json`.

## 1. Arithmetic

The report's Task 2/3 table reproduces exactly from `results/main/runs/` (5 cells × 5 seeds). The pass counts reproduce: 25/25 runs fall inside the roughness band and 0/25 pass the literal rule. 7/25 checkpoints are selected at the final update; 3 of these are whitened-MLP runs. The issues below concern interpretation, not arithmetic.

## 2. The residual group is mostly PC3, plus ten nugget directions

The report attributes the literal-rule failure to "anisotropic geometry". A more specific account follows.

- **PCs 7–16 carry no structure.** Their eigenvalues are about 5–6·10⁻⁶ for the target and 6–7·10⁻⁶ for the GP-source velocity. The kernel's numerical nugget is 10⁻⁶. Sundial normalization multiplies it by E[1/σ_h²] = 5.58 (Monte Carlo over 200,000 raw windows), giving 5.6·10⁻⁶. The source nugget adds another 10⁻⁶ to the velocity tail. So these ten directions are the stabilizing nugget, and they are the ten directions floored at 8.7·10⁻⁵.
- **The residual MSE is essentially the PC3 error divided by 14.** Mean GP+MLP raw errors are 0.597 (PC3), 0.084 (PC4), 0.0083 (PC5), 0.0005 (PC6) and about 10⁻⁵ for PCs 7–16. PC3 contributes 87% of the residual group.
- **Consequence.** The literal ratio residual/PC1 is mechanically small. Part of the 0/25 outcome comes from averaging one signal-bearing curvature direction with ten nugget directions. The geometry by itself does not explain it. The privileged oracle fails the same rule (residual/PC1 = 0.004).

## 3. Normalized residual errors are dominated by nugget directions

The GP+S4 normalized residual error of 25.1 is 92% PCs 7–16. There S4's raw error is 2.8·10⁻³, against 9·10⁻⁶ for MLP. The corresponding tail shares are 31% for MLP and 29% for the whitened MLP.

The controlled follow-up's raw-loss contrast (normalized residual 31.1 for S4, 0.222 for MLP, reported as its strongest finding) is 93% PCs 7–16 for S4. Its balanced-loss values (0.158 and 0.175) are less tail-dominated, at 28% and 45%.

These numbers do measure a real behaviour: S4 injects high-frequency energy of amplitude about 0.05 into directions where the target has none. They are not evidence about learning signal-bearing small components, and their magnitude depends on the nugget and the variance floor.

## 4. The roughness band cannot detect missing curvature

Project the seed-0 test targets (1,024 paths) onto k training-target PCs:

| k | variance lost | roughness ratio |
|---|---|---|
| 1 | 14.7% | 0.479 |
| 2 | 1.6% | 0.877 |
| 3 | 0.15% | 0.973 |
| 4 | 0.009% | 0.997 |

Paths with no curvature (k = 2) still pass the ±20% band. "25/25 roughness passes" therefore says little about path shape. The report notes a different limitation, oracle variance suppression, but not this one.

## 5. Head comparison against the pre-written branches

GP-source contrasts, S4 minus MLP, five paired seeds, with t intervals (4 df):

- **PC1 MSE:** +0.573, interval [−0.27, 1.42]. Not resolved. The report's percentile bootstrap, [0.05, 1.10], excludes zero (see §7).
- **Residual MSE:** +0.0090, interval [0.0077, 0.0103]. S4 is worse in all five seeds.
- **Roughness:** both heads pass in all seeds. Distance from target is 0.018 (MLP) versus 0.039 (S4) on average, and MLP is closer in 3 of 5 seeds.

On all three numbers the MLP therefore matches or beats S4. This meets the condition of the assignment's "Prior was overstated" branch. Every cell, S4 included, also fails the literal rule, which meets the condition of the "Empty task" branch. The assignment does not say which branch takes precedence when both hold. The report chose "Empty task", which is defensible, and this addendum does not relabel the primary outcome. It records that, judged on the head comparison alone, the evidence favours "Prior was overstated".

Later studies replicate the residual ordering, with MLP better:

- the lengthscale-6 study;
- the controlled follow-up, under both losses;
- the covariance-mixture study, on both datasets.

On forecast CRPS, S4 is slightly better on a pure GP source in the mixture study (0.487 vs 0.502). Forecast quality and the hypothesised mechanism point in different directions.

## 6. The whitening recommendation has no forecast-score basis in the report

The report calls whitening "a promising follow-up" on the strength of velocity MSE alone. It contains no proper scoring rule.

Later evaluations do not show a forecast benefit:

- **Original checkpoints on fresh histories:** CRPS 0.4663 (MLP), 0.4613 (S4), 0.4606 (whitened MLP). All intervals against MLP include zero.
- **Covariance-mixture study, GP endpoint:** full whitening gives 0.5244 / 0.4888 against 0.5152 / 0.4878 for MLP. Coordinates alone give 0.5045 / 0.4812.

Read the recommendation together with `docs/followup_conclusions.md`.

## 7. Five-seed intervals are optimistic

The report's paired intervals come from a percentile bootstrap over five seeds. Five values allow only 126 distinct resamples. For the whitening residual effect, the bootstrap gives [−0.00961, −0.00855] and the t interval gives [−0.01005, −0.00830]. For S4 − MLP on PC1, the bootstrap excludes zero but the t interval does not (§5).

Treat the intervals in `output/analysis/paired_seed_contrasts.csv` as narrower than warranted. They also reflect only optimization variance on one fixed data draw.

## 8. The "matched" source is narrower than the normalized target

The report says the source and target covariances are "not identical", without giving a size. The normalized target's variance exceeds the source's by a factor of 6.1, 3.7, 3.8 and 4.1 along PCs 1–4. In standard deviation the source is about 2–2.5× narrower.

No source matched in normalized coordinates was tested. The GP→GP cell therefore approximates, but does not realize, "same Gaussian to same Gaussian".

## 9. Parameter coverage

Every completed univariate run uses an RBF kernel with F = 16 and F/ℓ = 2 or 2.7. For RBF the geometry depends only on F/ℓ and the nugget, so the whole study covers essentially one geometry.

Raw-kernel spectra at F = 16, without normalization or training:

| kernel | F/ℓ | top-2 share | components > 10⁻³ of the first |
|---|---|---|---|
| RBF | 2 / 4 / 8 | 0.97 / 0.82 / 0.54 | 4 / 6 / 11 |
| Matérn-3/2 | 2 / 4 / 8 | 0.91 / 0.74 / 0.49 | 8 / 15 / 16 |
| OU (Matérn-1/2) | 1 / 2 / 4 | 0.88 / 0.77 / 0.61 | 16 / 16 / 16 |

The Gauss–Markov kernels relevant to the latent-prior programme (OU and half-integer Matérn) have power-law spectra and fail the 0.9 gate at F = 16. The design constraint the assignment asks for is thus measured only for a non-Markov, super-smooth kernel, in a regime with three or four meaningful components, where a temporal head has little to exploit. A separate pre-declared kernel/F/ℓ scan is being prepared.

## 10. Ordering claims cannot be checked against version control

The protocol, configurations and all main results were added together in commit `2758670` (2026-10-05). Three statements therefore rest on the team's account and the run logs rather than on version history:

- "declared before the scored experiments";
- "lengthscale 8 chosen before main training";
- "smoke checks did not influence choices".

## What this addendum does not change

The literal 20% rule, its 0/25 outcome and every saved metric stand as reported. The report's qualified recommendation is not withdrawn. The points above narrow how its velocity metrics, roughness passes and intervals should be read.
