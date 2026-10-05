# Independent model and optimization audit — 4 October 2026

This audit reads the actual implementations, model tests, saved five-seed primary checkpoints and nine optimization-sensitivity records. It does not modify earlier training code or results.

## Numerical findings

No substantive error was found in the real-valued rank-one S4 kernel, finite bidirectional convolution convention, velocity-coordinate transforms, or runner's paired random draws. The existing complex dense bilinear reference is independent of the optimized real inverse/Krylov calculation and tests every parameter gradient; the FFT reference separately tests temporal indexing. These are meaningful checks, rather than merely matching output shapes.

An additional check loaded **all 15 learned S4 kernels from the five GP-source primary checkpoints**, converted copies to float64 and compared them with the full conjugate-state complex bilinear reference. Maximum absolute disagreement was **8.40e-16** in float64; the largest difference between the original float32 CPU kernel and that reference was **7.81e-7**. Records are in `results/diagnostics/model_audit_trained_kernels.json`. This check uses CPU copies and makes no new claim about MPS determinism. Existing CPU/MPS forward/backward tests cover device arithmetic.

The initialization test's trace and input-norm invariants are weaker than proving full HiPPO similarity, but inspection of the construction agrees with the LegS formula. This is a test limitation, not an observed numerical defect.

## What the model comparison actually controls

Both models have the same lookback-encoder architecture and initial encoder weights, paired data/source/time draws, equal update budgets and nearly equal total trainable parameter counts. Their encoder weights then adapt independently. Consequently the experiment compares two end-to-end predictor families, not isolated heads receiving one fixed representation.

MLP input/output matrices give it direct absolute-position dependence. The S4 backbone has a shared scalar input projection, shared pointwise output and broadcast history conditioning; absolute position emerges only through finite-sequence boundaries. This is an architectural distinction, not an implementation error. Explicit position-conditioned S4 would test whether this distinction contributes to the observed gap.

The S4 implementation uses 16 real states, while the cited TSFlow backbone uses 128. A state-size control is warranted before attributing a small-model result to the full published architecture. Horizon 16 does not test S4's long-sequence scaling or memory advantages. Increasing the state dimension while holding width fixed also increases parameter count and must be disclosed or matched with an MLP size control.

The no-weight-decay rule for SSM transition/input/step-size parameters is honored. Equal global learning rates are a reproducible budget choice, not evidence that both families are equally optimized. The existing three-seed lower-rate warm restarts improve both families. They reset Adam and begin at each validation-selected checkpoint; they are a separate optimization sensitivity, not a continuation with preserved optimizer state or a convergence demonstration.

## What existing results already say

S4's GP-source residual velocity MSE exceeds ordinary MLP's on every one of the five primary seeds, and on every one of the three lower-rate warm-restart seeds. PC1 differences are smaller and the shorter-lengthscale experiment changes that metric's ordering. The current evidence therefore supports a narrow residual-error finding for these configurations; it does not support the expected blanket S4 advantage or a general MLP superiority claim.

## Prioritized falsifiable next experiments

1. **Equal loss preconditioning for both heads.** Keep input/output indexed by time, but evaluate the training loss after projecting velocity error through the training-fitted regularized PCA whitening matrix. Compare raw-loss MLP/S4 with weighted-loss MLP/S4 on fresh data. This asks whether residual error primarily reflects objective weighting. Applying S4 convolution directly over PCA coordinates is a different test: PC index is not temporal position, so full-coordinate-whitened S4 must not be presented as an unchanged temporal prior.
2. **Identical frozen encoder control.** Freeze the same deterministic representation for both heads and include a head-only parameter count. This separates representation co-adaptation from head differences. A random frozen encoder may impose its own bottleneck; an identity or explicitly fixed informative representation should be justified.
3. **S4 position/state sensitivity.** Add an explicit fixed normalized position channel, or increase state dimension, one change at a time. Pair fresh data and optimization seeds; match parameters where feasible; preserve the earlier results. Report if neither intervention helps.

Choose this bounded protocol before seeing new test outcomes. Validation selects checkpoints; fresh test draws avoid repeatedly adapting experiments to the original test set. Report every planned cell and distinguish targeted exploratory follow-up from the assignment's original primary comparison.
