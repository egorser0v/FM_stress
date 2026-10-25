# F64 capacity and budget controls

Frozen before scored training on new draws. This supplementary study follows the exploratory discovery that the original width-56 MLP at horizon 64 has a fixed output space of dimension at most 56. The same architecture at horizon 16 has no such dimension bottleneck. No original result or Task 1–3 flag is replaced.

## Questions and design

1. Does S4's white-source residual-error advantage survive removal of the MLP output-rank bottleneck at equal parameter count?
2. Do conclusions depend on a 6,000 versus 12,000 update budget?
3. Is this specific to smooth RBF targets, or also present with OU targets?

36 fresh MPS fits: RBF × {MLP width56 depth3, MLP width65 depth2, MLP width72 depth3, S4 width59 depth3} × {white, GP} × three draws; OU × {MLP width65 depth2, S4 width59 depth3} × both sources × three draws. Data seeds 607041/707041/807041, paired optimization seeds 61/62/63. Every geometry uses lookback128, horizon64, lengthscale8, variance1, nugget1e-6 and history-only Sundial normalization. The GP source is the raw kernel drawn independently in normalized coordinates, not the actual normalized target covariance.

The original MLP has 73,960 parameters; width65 depth2 has 73,953; S4 width59 has 74,038; the additional same-depth width72 MLP has 114,328. Width65 clears the guaranteed exact local rank cap below64 from final LayerNorm (width64 would still cap it at63). Finite-epsilon LayerNorm can still attenuate an additional radial direction, so width65 may have practical numerical rank63; the width72 control clears that potential bottleneck as well. Matching parameter count changes depth, so the larger same-depth control is needed. It is not itself a parameter-matched S4 comparison.

Training reuses official adapted models and the kernel-scan protocol: 32768/4096/8192 independent train/validation/test windows, batch256, constant AdamW learning rate1e-3, clip1, weight decay1e-4 with SSM exclusions, raw velocity MSE, independent coupling. Every fit runs12000updates. Separate best-validation checkpoints are retained over steps≤6000 and≤12000, evaluation every500. Both are evaluated only after all12000updates. These are paired budget observations from one trajectory, not72 independent trainings. No test-based checkpoint or hyperparameter selection. Encoder initialization, minibatch indices, source normals and times are paired and hashed across heads. CUDA historical results are reference only; primary comparisons use fresh MPS runs together.

## Evaluation

Original raw PC1 MSE, residual-PC3:F mean MSE, roughness and literal success rule; PC2 separately. Fair CRPS and energy score use256histories×32generated draws, Euler64; first draw repeats first8histories atEuler128. Sources have different velocity-noise floors, so raw-MSE comparisons are within-source; fixed train-target PCA metrics and the privileged raw-history Gaussian reference are also reported. The latter has more information than the model and is not an attainable model-specific oracle.

For every MLP, SVD of the trained output weight yields its orthogonal complement Q. The projection of velocity there equals the projection of the fixed output bias at all inputs. Consequently projected source variation survives the entire Euler trajectory: Q^T generated = Q^T source + Q^T bias. Report null dimension, forced squared velocity error, source and target projected variance, and numerical identity residuals. This exact global architectural fact does not rely on optimizer convergence. All models additionally receive local CPU Jacobian singular-value probes at flow times .1,.5,.9 along the actual interpolation of one held-out source/target pair and its history; these are descriptive and their numerical rank is threshold-dependent.

All contrasts are paired by independent draw. Report each draw and their average; three draws do not support broad statistical claims. RBF and OU share random-number seeds intentionally, not independent replications. The wide-model and altered-depth controls distinguish only some architectural causes. Fixed budget and checkpoint boundary counts must be disclosed. Tests and software smoke use separate data seed907041; smoke outcomes cannot influence scored hyperparameters.

## Execution note

Independent code review before any scored test output corrected a configuration mismatch in solver subset size and changed Jacobian probes from a fixed state to the corresponding interpolation state at each time. One partially trained initial worker was interrupted before held-out scoring; its logs, configuration, manifest and checkpoints remain under `results/followup-capacity/pre-audit-interrupted`. It supplies no results. Training settings and the36fit matrix were unchanged. The finalized study restarts from scratch.
