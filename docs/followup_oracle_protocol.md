# Information-matched oracle follow-up

Frozen design before model probes on 25 October 2026. This is a supplementary,
post hoc diagnostic of already trained models; no retraining or model selection.

## Question

The earlier Jacobian reference conditions on raw history, including its mean and
scale. Networks observe only its centered, standardized shape. Does comparing
against the Bayes field with the **same observed history** change the MLP/S4
ordering? It can correct an information mismatch; it cannot establish a causal
optimization mechanism or prove that an encoder learns the entire history.

## Derivation

Write raw history H = m 1 + s z, where z has zero mean and population variance
one. Its change-of-variables factor is s^(L-2). For P = K_hh^-1, define
c = 1'P1, b = 1'Pz, a = z'Pz - b²/c, A = K_yh P, d = A1 - 1 and
V = K_yy - A K_hy + dd'/c. Integrating m gives
s² | z ~ Gamma((L-1)/2, rate=a/2), truncated to s>0.01, and
normalized Y | z,s ~ N(Az - d b/c, V/s²). Without truncation this is a
Student-t with L-1 degrees of freedom and scale matrix a V/(L-1).
The scale matrix differs from covariance a V/(L-3).

The threshold branch is observable: normalized-history std <=0.01 implies the
normalizer used scale=1. The centered raw history is then known exactly;
normalized Y has Gaussian mean Az - d b/c and covariance V. No histories or
outcomes are silently excluded.

For independent Gaussian source, each mixture component's x_t is Gaussian.
Its likelihood reweights the scale posterior. The velocity is the posterior
weighted Gaussian conditional velocity; its Jacobian includes the covariance
of component velocity with component likelihood score. An average of component
Jacobians alone is wrong. Float64 generalized eigendecomposition of (V,C_source)
reduces integration to scalar diagonal variances. No artificial covariance floor.

## Frozen evaluation

- All 25 primary checkpoints, existing five optimization seeds, CPU.
- Reproduce the earlier probe geometry: first 256 original test histories,
  source normal seed424242, t in .1,.3,.5,.7,.9; same sources and training-PC bases.
- Additional velocity-only assessment: 2048 fresh independent GP histories,
  data seed9251041, source seed9251042 and uniform-time seed9251043.
- Report each seed/time, per-PC gains, absolute Jacobian errors, mean relative
  errors and pooled-norm relative errors against both information-matched and
  privileged references. Ratio averaging can be unstable near zero, so retain
  numerator and denominator norms. Separate PC1–2 and PC3–16 blocks.
- Velocity diagnostics in raw coordinates: label MSE, model-oracle discrepancy,
  oracle label MSE, expected conditional label noise, and exact signed cross-term
  decomposition. Teacher-forced errors are not generated forecast quality.
- Oracle target means/covariances independently checked using raw-history
  conditional sampling. Verify Jacobians by centered finite differences.
- Quadrature32/64/128 convergence; adapt to256 on discrepant records and report
  unresolved errors rather than silently declaring exactness. The scale tail is
  dropped only when posterior probability<=1e-12; otherwise use truncated-Gamma
  quantile quadrature and report that branch. Threshold probabilities and all
  numerical diagnostics are recorded.
- Hash configuration, implementation and checkpoints before model evaluation.

## Interpretation

A residual-Jacobian ranking that survives the matched reference is evidence of
field-response mismatch for these checkpoints. It does not identify why it
arose. The oracle sees full normalized history; the model's tiny learned encoder
may discard information. This is a Bayes reference for network inputs, not its
bottleneck representation. The Student-t field is generally nonlinear in x_t.
The 256 old contexts are post hoc; 2048 fresh contexts do not create independent
training replications. The assignment's three original metrics and success rule
are unchanged. No claim about published TimeFlow or TSFlow is licensed.

## Explicit post-hoc secondary addition

After the first completed oracle probes, the reference correction left S4 closer
in unweighted Jacobian norm despite its worse residual velocity errors. Added
covariance-weighted local Jacobian discrepancy to examine this metric mismatch.
For a perturbation with covariance Cov(x_t | normalized history), measure
trace((J-J*) C_x (J-J*)'). Main/residual blocks restrict both inputs and outputs
and use the corresponding covariance submatrix. Both references use the **same**
normalized-history input covariance. This diagnostic weights typical directions;
it is a local linear sensitivity measure, not actual forecast error and not a
new primary endpoint. Its post-hoc status is recorded in the run configuration.
