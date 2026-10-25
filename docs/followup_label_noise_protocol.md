# Conditional label-noise intervention

Frozen before scored training on new data, 25 October 2026. Motivation follows
our information-matched oracle audit. This is a causal intervention on the
training target estimator, within the original F16 RBF GP setting.

## Design

24 fits: white/GP source × MLP/S4 × ordinary independent-coupling velocity label
or conditional-mean oracle label × three independent data draws. Data seeds
917041/1017041/1117041 and optimization seeds81/82/83. L32 F16 ell8, variance1,
nugget1e-6, history-only Sundial normalization. Train32768, val4096, test8192.
Both heads retain width56, depth3, same tiny encoder architecture/initialization.

A precomputed full stream has6000×256 distinct pair draws: training-history
index, independent Gaussian source, uniform flow time. It is identical across
heads and label modes, with matched source normals across white/GP. Store x_t,
u=y−epsilon, and the information-matched conditional mean E[u|x_t,h_normalized].
The latter analytically integrates missing raw history mean/scale; it does not
condition on the sampled raw history. 64-point quadrature is checked against128
on held-out portions of the stream and complete validation/test pairs. Memory
maps avoid loading multiple streams into device memory. The stream never repeats
by design, although history indices recur. No training data from other studies.

6000 AdamW updates, batch256, lr0.001 constant, weight decay0.0001 with SSM
exclusions, gradient clip1. The validation criterion for BOTH label modes is
conditional-mean oracle MSE, every500steps. This deliberately removes validation
label noise and differs from the original baseline's selection policy. No
training/test-dependent setting or checkpoint tuning. Model initialization and
training inputs are paired. An interrupted unfinished fit restarts fresh.

## Interpretation and endpoints

Conditional expectation preserves the population regression optimum and removes
conditional label variance. A fixed finite empirical history pool is reused, so
it is not an exact statement about the empirical paired-label objective. This
study measures the effect of providing noiseless conditional-mean supervision,
not a deployable technique; the generator and analytical oracle are privileged
knowledge, while the oracle's conditioning variables match network inputs.

Primary descriptive contrasts: oracle/stochastic supervision ratio for raw
residual MSE and matched-oracle residual discrepancy, paired within source/head;
interaction of that improvement with source and with head. Report every draw,
then equal-weight means. Three data draws limit generality. Improvement supports
label-variance-limited learning at this architecture/budget/selection policy;
it does not prove an intrinsic architecture limitation or ill-conditioned Hessian.

All three original assignment metrics and literal20%rule remain reported. Also
CRPS/energy/calibration on256histories×32draws at Euler64, same test ensembles
across label modes/heads; Euler128 first8histories of first draw. Jacobians at
128held-out interpolants at t=.1,.5,.9: compare both equal-direction and
conditional-input-covariance-weighted error against the matched oracle. These
are explanatory metrics, not new assignment success criteria. Record boundary
checkpoint selections, source/config/checkpoint hashes, all predictions/samples,
quadrature convergence, and paired stream/encoder hashes.

Software tests and smoke use separate data seed9917041 and four updates. No
scored run begins until independent implementation review. Results and reports
are separate from earlier artifacts. No commits.

## Pre-training implementation review

Preparation computes validation and test reference arrays and checks analytic
quadrature before training. Test arrays/scores never enter optimization,
validation selection or setting selection. They are final-scoring inputs, not
untouched files first accessed after fitting. Float64 oracle labels are stored
as float32; first/last stream probes recompute the teacher at the actual rounded
x,t,h inputs, including residual/tail-PC discrepancies. A relative RMS above1e-4
aborts preparation. Prepared arrays are hash-verified once per process/group
before training. Full runs require actual MPS with no silent CPU fallback;
finite training loss, gradient norm and validation score are enforced.
