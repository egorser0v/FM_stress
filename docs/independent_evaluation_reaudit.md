# Independent evaluation and statistical re-audit

Re-audit on 2026-10-04 of the supplied four-page assignment, `data.py`,
`metrics.py`, `experiment.py`, the tabulation/contrast logic in `analyze.py`,
protocol, existing final audit, optimization sensitivity and saved GP-cell run
records. This audit concerns evaluation and design; a separate reviewer should
check S4 implementation and accelerator numerics. No original numerical source or
result was modified.

## Findings

No blocking error was found in the inspected data generation, CFM construction,
PCA projection or primary metric calculations. Histories and futures are drawn
jointly; train/validation/test use distinct streams. Source draws are independent
of targets, training uses fresh source and time draws, and all heads within a
source family see the same fixed evaluation pairs. The independently reset
training stream prevents architecture initialization from changing subsequent
sampling. Whitening consistently maps input patches and derivative labels, then
maps predictions back before original-coordinate evaluation. Errors are projected
without subtracting their test mean.

The saved GP-source checkpoints for all five main seeds and all three shorter-
lengthscale seeds select the first minimum of the recorded validation objective.
The independently recalculated residual-MSE means are 0.04927195 / 0.05826895 /
0.04009735 (main MLP / S4 / whitened MLP), and 0.12372252 / 0.13109898 /
0.10859188 (shorter-lengthscale series), matching the final report's conclusions.
These numerical comparisons do not establish convergence or universal superiority.

The following qualifications remain necessary:

- The supplied assignment leaves the normalization and directionality of “within
  20%” underspecified. The project declares a literal two-sided, per-component
  raw-error interpretation before the scored runs. This is a transparent
  operational choice, not a uniquely mandated mathematical interpretation. Its
  failure can coexist with better residual accuracy. Do not use the failure flag
  to claim the architectures cannot model smooth paths.
- Source covariance matches the **raw** data kernel. Random history normalization
  changes the target marginal distribution and covariance. This is disclosed;
  “identical normalized source and target distributions” would be false.
- The exact Gaussian oracle receives raw history while learned networks receive
  normalized history. Its errors are a privileged diagnostic, not the exact
  attainable risk for the networks. Do not interpret a model–oracle gap entirely
  as optimization failure.
- Independent CFM labels have irreducible conditional variance. Label MSE does
  not directly measure error of the learned population vector field. Within-
  source, paired head comparisons remain meaningful because label distributions
  and PCA bases are held fixed.
- The fixed raw objective is comparable between MLP and S4; the whitened model
  selects checkpoints with a different objective. That is part of the declared
  coordinate intervention. Its benefit cannot isolate input coordinates from
  loss weighting or checkpoint selection.
- One generated path per history and aggregate adjacent-jump roughness do not
  evaluate conditional uncertainty or forecast calibration. Similar roughness can
  coexist with bad variance, bias or covariance. Existing oracle Euler checks
  demonstrate this possibility, but do not diagnose learned-model calibration.
- Five optimization seeds share the same data and evaluation draw. Their paired
  bootstrap intervals describe optimization variability conditional on that draw.
  They are not independent dataset replications or reliable general-population
  significance tests. The numerous unadjusted contrasts are exploratory.
- The analysis exports cross-source velocity-MSE contrasts. Those quantities
  compare different labels and PCA bases and are not clean forecast-quality
  effects of source choice. Existing narrative correctly avoids this inference;
  any future presentation must retain that warning.
- Shorter-lengthscale and warm-restart studies followed inspection of earlier
  outcomes. Their supplemental labels and validation-only warm-restart expansion
  rule are appropriate. They must not be retroactively presented as independent
  confirmation of an unseen hypothesis.

## Recommended prospective extensions

The highest-value next measurement is conditional sampling on a new untouched
synthetic test draw, using existing validation-selected checkpoints. Before
running it, fix the number and seed of histories, source ensemble size, Euler
steps, included heads/seeds and reported metrics. For example, use 256 or 512
histories, 64 independent source draws per history, and Euler-128. Include every
available main seed for GP+MLP, GP+S4 and GP+whitened MLP. Use identical standard-
normal draws and histories across comparable heads. Score ensemble CRPS for
marginals, energy score for entire paths, central interval coverage and width,
predictive-mean error, and generated roughness. The observed target future is a
valid outcome for proper scores even when the normalized-history conditional law
is unavailable analytically. Bootstrap histories as clusters; do not treat future
coordinates or ensemble members as independent cases. Report variability across
training seeds separately. A higher-step sensitivity on a predetermined subset
can distinguish solver effects from head differences.

For new training, compare a prospectively fixed small optimizer/state-size search
with equal search budgets for both architectures. Select settings only on fresh
validation data, then evaluate a new test draw once. A balanced per-PC loss while
keeping **raw time-domain inputs** for both models is a particularly interpretable
control: it can test objective conditioning without feeding S4 a sequence of PCA
indices that no longer represents time. Directly whitening the S4 input is not a
clean test of its temporal inductive bias. A supplementary analytic-teacher study
can remove label noise, but must give both heads the same information sufficient
for that teacher and explicitly acknowledge the changed task.

A useful result is a bounded empirical comparison, including a null or conflicting
ranking. The extension must not keep changing its geometry, metrics or tuning
until S4 wins. Existing primary results and their success flags should remain
unchanged.
