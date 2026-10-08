# Sparse models inside conditional flow matching

These are new small implementations for this project's controlled extension.
They are **not reproductions** of the reference papers' complete training systems,
pretrained models, or reported forecasting results. Existing model files remain
unchanged. All models receive an interpolated future `x_t`, scalar flow time `t`,
and observed history `h`, and return a joint `[batch, horizon, channels]` velocity.
They train with the same conditional flow-matching objective and generate by ODE
integration. The autoencoders are internal conditioning components, not separate
forecasting methods.

## API and loss

`create_sparse_model(head, lookback, horizon, channels, width=56, depth=3,
latent_dim=64, topk=8, conv_steps=3, conv_kernel=5,
reconstruction_weight=0.05, sparsity_weight=0.001, l0_weight=0.001,
gate_seed=1729, time_dim=32)` constructs the requested model.

Call `prediction = model(x_t, t, history)` then
`loss = mse(prediction, velocity_target) + model.auxiliary_loss()`.
The history AE auxiliary terms are cached by the last forward; consume the loss
before another forward. The optional `history` argument to `auxiliary_loss` is
accepted for compatibility but is not recomputed. Cache entries are replaced,
not appended. `activity_statistics()` returns detached Python scalar diagnostics.
These diagnostics describe the **latest batch**, not a whole split unless the
caller aggregates them. No auxiliary loss is added to generation or scoring.

## History TopK autoencoder and dense control

`history_topk_ae` computes a ReLU code from the flattened observed history, keeps
at most `topk` nonzero coefficients, and feeds the code into the shared AdaLN FM
network as conditioning. A linear decoder reconstructs **only that history**.
The learned decoder atoms have unit L2 norm. Auxiliary loss is
`0.05 * mean((reconstructed_history - history)^2)` with default settings.
`history_dense_ae` is identical, including initialization and parameter count,
but omits TopK. There is no pretrained language-model SAE or future-target
reconstruction. Sparse conditioning is trained jointly with flow matching.

The architectural idea follows Gao et al., *Scaling and evaluating sparse
autoencoders* (2024): [paper](https://arxiv.org/abs/2406.04093),
[official code](https://github.com/openai/sparse_autoencoder). This adaptation
omits the original system's dead-latent auxiliary machinery and large-scale
training procedure. A fixed small training budget may leave inactive atoms;
report actual code activity and reconstruction quality rather than assuming all
latent features are useful.

## Convolutional sparse history representation

`history_conv_sparse` uses three unrolled proximal-gradient steps with one tied
multichannel convolutional dictionary. Each step computes the residual history,
applies the adjoint analysis convolution, and soft-thresholds signed codes.
Thresholds are learned positive values; learned steps lie in `(0,2)`. The
convolution is normalized by a Young-inequality upper bound on its operator norm,
so its squared norm is at most one. A transpose convolution reconstructs history.
Four ordered temporal pooling bins convert the code to AdaLN conditioning.
The auxiliary loss adds `0.05 * reconstruction_MSE + 0.001 * mean(abs(code))`.
These coefficients are configuration defaults, not validated optimal choices.

`history_conv_dense` runs exactly the same iterations and conditioning pipeline
without thresholds or the code L1 penalty. It has the same widths, dictionary,
and reconstruction loss. The sparse model has `latent_dim` additional threshold
parameters. This is a CRsAE/ISTA-inspired history encoder, not an exact CRsAE
reproduction: it does not implement that paper's FISTA acceleration, original
noise model, or experimental protocol. Reference:
[CRsAE author repository](https://github.com/btolooshams/crsae).

## Sparse velocity dictionary

`dictionary_topk` predicts signed coefficients from `(x_t,t,h)`, retains at most
`topk` coefficients by absolute magnitude, and linearly combines unit-norm
learned dictionary atoms into velocity in the original output coordinates.
`dictionary_dense` uses the identical architecture with no truncation. Neither
has a reconstruction loss. The label is **TopK velocity dictionary head**, not
SAE: this model does not encode and reconstruct a separate input. Sparse codes
can suppress weak but useful temporal components; evaluate the original PC-wise
errors and complete forecast scores, not just support size. All arrays use joint
time/channel dimensions, never channels-as-independent-batch.

## Hard-concrete U-Net channel gates

`unet_l0` is the existing temporal U-Net with a trainable hard-concrete gate after
each encoder and decoder block. A gate is shared across the batch and time axis;
channels are gated, not individual weights. Training samples stochastic gates;
evaluation uses the standard deterministic stretched sigmoid. The auxiliary
term is `0.001 * mean(expected_nonzero_probability)` over every hidden gate.
This is a normalized expected channel count, not a weighted parameter/FLOP cost.
Initial logits are zero (unstretched sigmoid 0.5). Weak regularization or a short
budget may produce no exactly zero deterministic gates; report expected activity,
actual evaluation activity, and mean gate amplitude separately.

Reference: Louizos et al., *Learning Sparse Neural Networks through L0
Regularization*, ICLR 2018:
[paper](https://arxiv.org/abs/1712.01312),
[official implementation](https://github.com/AMLab-Amsterdam/L0_regularization).
The new implementation uses temperature `2/3` and stretch interval `[-0.1,1.1]`.
Each gate has a private CPU generator, seeded independently and saved via
`get_extra_state` in checkpoints. Forward calls do not alter the global PyTorch
RNG used for paired training examples, sources, or interpolation times.
Evaluation neither samples gates nor advances their RNG. Callers serializing
state dicts must retain non-tensor extra-state dictionaries.

## Interpretation and limits

Dense and sparse alternatives within each pair must use identical widths and
training budgets. Match parameter budgets **between families** without changing
widths inside the dense/sparse comparison. Auxiliary reconstruction introduces a
second learning objective; the dense AE control receives the same objective.
Convolutional sparsity changes both thresholding and the L1 auxiliary penalty,
so their joint effect is evaluated unless separately ablated. L0 changes both
stochastic training and regularization, not solely final channel count.

All models use ordinary dense PyTorch operations. Sparse activations and gates
are not evidence of faster MPS kernels, reduced parameter allocation, or lower
end-to-end runtime. Parameters are still stored and optimized densely. These
heads change the original model assumptions; measured improvements must be
reported as extension results, without rewriting the original GP hypothesis.
