# Protocol: smooth-source conditional flow matching

This protocol makes the two supplied project PDFs operational. The primary task is a
small synthetic stress test, not a reproduction of either pretrained Sundial or the
complete TSFlow forecasting system. All model results must be labeled by the actual
architecture and training budget used. No general claim about TimeBench is supported.

## Data and fixed geometry

The default is a zero-mean squared-exponential Gaussian process on a unit-spaced
joint window of length 48: a 32-point history and a 16-point future. The covariance
is `exp(-(i-j)^2/(2*8^2)) + 1e-6*I`. The nugget is part of both raw-data and GP-source
covariance, not a hidden sampling approximation. An independent window is drawn for
each example; there are no overlapping segments of a common trajectory. Train,
validation, and test use disjoint RNG seeds. Test draws are never used to fit PCA,
select lengthscale, choose checkpoints, or choose a model.

The lengthscale 8 choice was informed by an explicit preliminary geometry scan,
before model training. It must not be described as a preregistration made before
any data were seen. That pilot showed target/white-velocity/GP-velocity top-two
energies about .979/.845/.977. These pilot estimates are not the final held-out
geometry table. Longer lengthscales can inflate normalized future variance enough
that even white-source velocities have top-two energy above .9. This is a reason
to inspect the contrast, rather than assume that a longer lengthscale is always
better. A scan over `[4,6,8,10,12,16,24]` is a geometry sensitivity analysis.

### Exact Sundial normalization

For each raw history `r`, compute `m = mean(r)` and the **population** standard
deviation `s = std(r, ddof=0)`. Set `s=1` if the standard deviation is at most .01.
Use `(r-m)/s` as history and `(future-m)/s` as target. No future value contributes
to the statistics. This is the released Sundial `revin=True` behavior, not
normalization of each future patch separately and not division by `std+epsilon`.

Verified source: [official model implementation](https://huggingface.co/thuml/sundial-base-128m/blob/main/modeling_sundial.py),
`SundialForPrediction.forward`, normalization at lines 397–418 in the version
retrieved 2026-10-04. The package's provenance record should pin the downloaded
revision used for architectural adaptation.

### Independent sources and what “matched” means

White source: `epsilon ~ N(0,I_F)`. GP source:
`epsilon ~ N(0,K_F)` using the **same kernel, lengthscale, variance, and nugget** as
the raw GP generator. Draw sources independently from target windows and use
separate RNG streams. Both sources are expressed directly in normalized
observation coordinates. Do not normalize a source using its paired target's
history statistics; that would create dependence through the history.

After the nonlinear, random context normalization, the target's unconditional
covariance is no longer exactly the original GP kernel. Thus “matched kernel”
means matching the raw generator, not claiming identical normalized marginals.
Both remain smooth. This distinction must accompany interpretations of prior
choice. Empirical covariance matching could be a separate ablation, but would not
replace this declared source midway through the experiment.

### Geometry gate before training

Fit separate PCA bases to training targets, white-source velocities and GP-source
velocities. Report held-out variance in the fixed train-fitted axes: top-one and
top-two shares, full spectrum, total variance, mean within-patch standard deviation
and mean adjacent absolute jump. Re-centering held-out values to estimate variance
does not refit axes. The matched-GP velocity must have top-two share >.90 on
validation. White-source energy <.90 is an additional check that the proposed
contrast is present. Nonzero within-patch variation and visually nonconstant sample
paths rule out an empty constant-series task. Thresholds and the selected geometry
must be recorded before training, and test spectra reported after selection.

## Models and training comparison

The required grid is white/GP source crossed with AdaLN MLP/three-residual-block S4
head, with a common small history encoder architecture and matched training
budgets. Use the same initial encoder parameters for paired cells and independent
model copies; if encoder parameters are not shared/frozen, state that the encoder
*architecture and initialization*, rather than final weights, match. Record total
parameters, steps, batch size, optimizer, learning rates and wall-clock time. If a
CNN or diagonal SSM is used instead of full S4, label the substitution explicitly;
a proxy cannot silently be presented as the assigned S4 result.

For every sample, independently draw `t~Uniform(0,1)` and `epsilon`, set
`x_t=(1-t)*epsilon+t*y`, and regress `u=y-epsilon`. The raw-cell loss is observation-
coordinate mean squared error. The model takes `x_t,t,encoder(history)`. Hold out
validation/test windows. Use common evaluation source/time draws within each
source family to reduce paired comparison noise. Training seeds must match across
cells; report individual seeds and mean/spread, not just a favorable run.

Generate from the declared source by explicit Euler integration with fixed `K`,
using `t=k/K` and `x += v(x,t,h)/K`, for `k=0,...,K-1`. Use identical `K` in the
primary table; check larger K separately to distinguish solver error from model
error. All paths and all reported primary metrics are in normalized observation
coordinates, including the whitened control's outputs after the inverse map.

## Three required metrics and the literal success rule

Let `q_j` be a train-fitted velocity PCA direction and `e=v_theta-u` the velocity
error on held-out independent couplings and uniformly sampled times. **Do not
subtract the test mean error:** that would erase prediction bias.

1. **Common-mode MSE:** `M1=mean((e dot q_1)^2)`.
2. **Residual MSE:** `MR=mean_j=3..F mean((e dot q_j)^2)`. This averages, rather than
   sums, the F-2 residual coordinates so its units are comparable to one PC.
   Report PC2 MSE separately; omitting it from the required residual metric follows
   the assignment but must not hide its error.
3. **Generated-path roughness:** `R=mean(abs(x[i+1]-x[i]))`, averaging over the
   F-1 adjacent pairs and every generated path. Report `R_target` on held-out
   target patches and `R/R_target` alongside the generated value.

Primary “works” is the literal two-sided conjunction
`0.8 <= MR/M1 <= 1.2` and `0.8 <= R/R_target <= 1.2`. Undefined ratios from zero
reference values are recorded as undefined and not declared successes.

**Important limitation:** the assignment's equality rule is not a general
accuracy criterion. A model with much *smaller* residual error can fail it. Raw PC
variances are highly unequal, so the raw ratio alone cannot establish that one
architecture is poorly conditioned. Also, `u` is a random conditional-flow label;
a perfect regression field can have nonzero label MSE. These limitations must be
reported even if all cells fail. The literal rule is never changed after results
are seen. A one-sided “residual no worse than 1.2 times PC1” interpretation, if
shown, is explicitly secondary, not a renamed primary success.

### Secondary scale and noise diagnostics

Report each PC's MSE divided by its **training velocity variance**, floored at
`max(1e-6*lambda_max, 1e-8)`. Report normalized PC1, PC2 and the average normalized
PCs3..F as well as raw metrics. This unitless score diagnoses relative error across
energetic and small-variance directions. Record the floor, eigenvalues and number
of floored directions. Applying the same 20% rule to these normalized scores is a
secondary diagnostic and must not replace the primary literal rule. PCA whitening
uses the same predeclared floor; disclose its effect on nearly null directions.

## Exact GP diagnostic and its information advantage

Given a full **raw** history `r`, the raw future is Gaussian with the usual GP
conditional mean and Schur-complement covariance. Transform it with the context
statistics above to get `mu_h,C_h`. Given independent Gaussian source covariance
`C_e`, the ideal regression field for this privileged information is

`v*(x,t,r)=mu_h + [t*C_h-(1-t)*C_e] [t^2*C_h+(1-t)^2*C_e]^{-1}(x-t*mu_h)`.

The corresponding conditional label-noise covariance is

`C_h+C_e - [t*C_h-(1-t)*C_e] [t^2*C_h+(1-t)^2*C_e]^{-1} [t*C_h-(1-t)*C_e]^T`.

Report its projected raw MSE floors and, when useful, empirical model-oracle
squared differences. Exact conditional future draws provide a reference
roughness/distribution check independent of learned integration.

The learned model receives only normalized history. Its context does not retain
the raw mean/scale used by this oracle. Therefore this is a **privileged lower-bound
diagnostic**, not an exactly attainable oracle for the model and not grounds to
attribute every excess error to optimization. Unconditionally, normalized future
patches form a mixture; treating their global empirical covariance as an exact
conditional Gaussian oracle is invalid.

## Task 3: mandatory control and coordinate audit

Fit PCA on **training GP velocities**. Let `Q` contain directions and
`D=sqrt(max(lambda,floor))`. Use the same AdaLN head and encoder architecture in
whitened coordinates. Transform patch, velocity and source consistently. A simple
linear map is `z=D^{-1}Qx`, target `u_z=D^{-1}Qu`; integration output maps back with
`x=Q^T D z`. A shared constant affine patch offset is also valid because its
time derivative is zero, but velocity must not subtract that offset. In particular,
subtracting the fitted mean velocity from the velocity target changes the flow
unless an explicit time-dependent correction is included. Score mapped-back
velocity and generated paths with exactly the same train-fitted GP PCA and three
metrics as the raw GP cells.

The control is run regardless of whether the initial MLP loses. Report whether it
closes an observed gap, changes solver sensitivity, or merely changes one of the
metrics. Do not infer a coordinate-cause story without an actual initial gap.

## Conclusions and deliverables

Required outputs are the one-page critical landscape note (Sundial, TSFlow, S4,
missing GP+MLP cell), held-out geometry table, four-cell metric table plus whitening
row, generated/target paths for each cell, and a recommendation paragraph. Release
configs, seeds, trained checkpoints, raw metrics, PCA arrays, training curves,
source provenance and tests so the result is reproducible.

If all four cells fail the literal success rule, follow the assignment's empty-
task branch: audit data/training, try a declared shorter lengthscale or longer
patch, rerun Task 1 before any rerun training, and withhold an architecture verdict.
An additional geometry is a labeled robustness/repair experiment; it does not
rewrite the primary experiment. If failure is due to the equality rule itself,
show the oracle/normalized diagnostics and explain that the project as written
does not support its intended method conclusion. Report a null or inconclusive
result rather than manufacture a winner.

The real-series extension is optional. Skip explicitly unless its instance-
normalized geometry passes the same test. Do not expand to a benchmark or imply
that this narrow GP geometry establishes general superiority of MLP, S4, whitening,
or a prior.
