# Sparse follow-up: sources and adaptation boundaries

Primary sources checked on 8 October 2026. These references motivate new small
conditional flow-matching experiments; they do not supply pretrained forecasting
heads interchangeable with the existing project.

| Method | Primary paper and author code | What transfers to this experiment |
|---|---|---|
| TopK sparse autoencoder | Gao et al., [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093); [official code](https://github.com/openai/sparse_autoencoder), MIT | Keep a prescribed number of active latent coefficients. Train a history reconstruction auxiliary task alongside conditional FM, and compare against the same dense history autoencoder. The original work studies language-model activations; our history encoder is a new adaptation trained from scratch. |
| Convolutional sparse coding / CRsAE | Tolooshams, Dey and Ba, [Scalable Convolutional Dictionary Learning with Constrained Recurrent Sparse Auto-encoders](https://arxiv.org/abs/1807.04734); [author repository](https://github.com/btolooshams/crsae), MIT, Keras and PyTorch implementations | Temporal convolutional dictionaries and sparse codes are a useful inductive bias. CRsAE specifically uses constrained recurrent, unrolled sparse inference. A small feed-forward convolutional TopK autoencoder must be called a convolutional sparse-autoencoder adaptation, **not a CRsAE reproduction**. |
| Learned L0 gates | Louizos, Welling and Kingma, [Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312); [author code](https://github.com/AMLab-Amsterdam/L0_regularization), MIT | Hard-concrete gates provide a differentiable expected-L0 penalty. Applying gates to temporal U-Net channels is a new architecture adaptation. Report the actual deterministic gates and expected active fraction separately; do not infer acceleration from zeros in dense kernels. |
| Sparse expert routing | Shi et al., [Time-MoE](https://arxiv.org/abs/2409.16040); [official code](https://github.com/Time-MoE/Time-MoE), Apache-2.0 | Input-dependent expert routing is a different kind of sparsity. A tiny FM expert head is a project adaptation, not a reproduction of the billion-scale autoregressive Time-MoE foundation model. Compare against all-expert routing at the same stored capacity and measure expert usage. |
| MultiTask Elastic Net | [Official scikit-learn estimator documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html) and [user guide](https://scikit-learn.org/stable/modules/linear_model.html#multi-task-elastic-net) | The L2,1 penalty jointly selects history lag/channel features across the flattened future outputs. We use the actual scikit-learn estimator, with 12 prespecified alpha/ratio pairs and chronological validation MSE selection. This is an external forecasting baseline, not a flow-matching model. |

## What the comparisons identify

- Fixed masked MLP: sparse weights, with the same mask used for every example.
- TopK history encoder: input-dependent activation sparsity, changing the past
  representation available to FM. A dense autoencoder controls for representation
  learning and the reconstruction auxiliary objective.
- TopK velocity dictionary: input-dependent sparse velocity coefficients. Without
  a separate autoencoding objective this is a dictionary head, not an SAE. Dense
  dictionary coefficients are its appropriate control. The decoder maps velocity
  coefficients directly to velocity; it does not decode latent positions as if
  they were derivatives.
- Convolutional sparse history encoder: combines local temporal filters with
  sparse activations. A dense convolutional autoencoder separates convolution
  from the sparsity intervention.
- L0 temporal gates: learned stochastic channel gates during training, with a
  deterministic prediction rule at evaluation. Training and inference sparsity
  are distinct quantities.
- MoE: sparse expert routing; dense stored experts do not imply sparse weight
  storage or faster execution when every expert is evaluated in a dense batch.

The original hypothesis concerns small-variance temporal structure. Sparsity may
discard precisely those components, so reconstruction MSE alone is insufficient.
Report history reconstruction error and latent activity alongside the original
velocity PC metrics, target roughness, CRPS and joint energy score of generated
forecasts. Every proposed neural encoder/head remains inside conditional FM.

## Whitening control and leakage boundary

The follow-up must also run MLP with full-rank PCA whitening on univariate and
multivariate targets. Fit the transform only on training velocities, consistently
transform source, interpolation and velocity, and invert final generated paths
before evaluation in the extension's common channel-standardized coordinates.
Keep a positive eigenvalue floor and report it. This differs from merely weighting
the temporal-coordinate loss by PCA variance; the previous balanced-loss control
did not whiten model inputs.

Use the same chronological splits, train-only channel scaler, evaluation context
indices, source draws and solver budget as the existing extension. No test score
selects a sparse model, a sparsity coefficient or a whitening floor. A generic
experimental implementation is not evidence that its source paper's results have
been reproduced.

The Elastic Net predictive distribution resamples whole training residual vectors,
matching the ridge baseline's residual draw seed. In-sample residuals may understate
uncertainty and their transfer to a later time period is an assumption.

## Exact QR acceleration for the linear baseline

The initial direct scikit-learn coordinate-descent run was stopped before producing
any fitted-dataset result because the full TRAIN design made the fixed grid slow.
The standalone `fm_stress/sparse_baselines_qr.py` preserves the same MultiTask
Elastic Net objective and 12 candidates. The frozen neural experiment code was
not edited for this numerical acceleration.

For centered training inputs `X = Q R`, split centered `Y` into its projection
`Q Q.T Y` and an orthogonal remainder. The remainder adds only a constant to
`||Y - X W||_F^2`. With `m = min(n_samples, n_features)`, fitting scikit-learn on
`sqrt(m/n) R` and `sqrt(m/n) Q.T Y`, without an intercept, exactly preserves
the original `1/(2n)` loss and both regularization terms. Restore the intercept
as `mean(Y) - mean(X) W`. No input features are rotated or dropped from the
coefficient penalty, and no validation/test information enters this operation.

Direct-solver parity is tested for tall and wide matrices. Each selected real
fit additionally reports primal and feasible dual objectives, the duality gap,
feature-group KKT residuals and the intercept gradient evaluated on the original
TRAIN arrays. Full-precision coefficients and residual bootstrap arrays are saved.
The baseline records its own source/configuration/data-metadata fingerprint and
sample checksum independently of the frozen neural experiment manifest.
