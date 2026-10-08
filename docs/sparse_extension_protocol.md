# Sparse representations and multivariate whitening: frozen exploratory protocol

This follow-up extends, rather than rewrites, the original assignment and the 96-fit extension. The assignment's rank-two geometry gate, success criterion and original results remain unchanged. The question is whether coordinate scaling, learned sparse representations or learned structural/routing sparsity improve conditional forecasting at a fixed small-model budget.

## Scope fixed before final training

Four datasets from `configs/extension.json`: independent three-channel GP, ETTh1 OT, joint seven-channel ETTh1 and joint eight-channel Exchange. Same chronological boundaries, contained windows, training-only global channel scaler, independent temporally white/GP source distributions with identical channel correlation, seeds 31/32/33, 2,000 updates, batch 128, AdamW LR .001, weight decay .0001, clipping 1, evaluation every 250 updates. No test-driven hyperparameter or model selection. A 50-update smoke pass checks numerical validity only and does not select hyperparameters.

11 new heads x 4 datasets x 2 sources x 3 seeds = 264 new neural fits. The previously completed 96 fits are frozen reference controls, hashed into the new manifest. In particular, original MLP and U-Net controls retain their widths. This is a fixed-budget exploratory comparison, not an exhaustive tuning competition or a convergence claim.

## Interventions and paired controls

- `mlp_whitened`: MLP of exactly the reference width; full-rank joint F*C PCA whitening, fit on training velocities independently for each source. For row vectors, W(x)=(x Q)/s, W^{-1}(z)=(z*s)Q^T. Position and derivative use the same linear map, with no derivative centering. Train on whitened velocity MSE; return velocity and forecasts to observation coordinates before evaluation. History is unchanged. Eigenvalue floor is max(1e-6 lambda_max,1e-8); save spectra and number floored.
- `mlp_balanced`: identical raw-coordinate MLP, but velocity errors are weighted by the same PCA inverse variances. Separates the coordinate-input/readout intervention from loss reweighting. Neither intervention truncates dimensions.
- `history_dense_ae` / `history_topk_ae`: the same trainable history autoencoder, latent width 64, decoder and conditional FM MLP; sparse version retains at most 8 latent activations. Both train history reconstruction with coefficient .05. Only history is encoded/reconstructed, never the unknown future. This changes conditioning; it is not a same-encoder head-only comparison.
- `history_conv_dense` / `history_conv_sparse`: three unrolled tied convolutional dictionary-inference iterations, kernel 5; sparse version uses learned soft thresholds and code L1 coefficient .001. Both use history reconstruction coefficient .05. This is an independently implemented CRsAE-inspired adaptation, not a reproduction of its full EM training procedure. The contrast bundles thresholding plus an L1 penalty.
- `dictionary_dense` / `dictionary_topk`: conditional velocity predicted using an overcomplete/learned linear dictionary with 64 coefficients, of which at most 8 are active in the sparse version. The state and history are available to the coefficient predictor. This is a sparse velocity head, not an autoencoder and not latent-space FM.
- `unet_l0`: the same U-Net width as the reference, with learned hard-concrete hidden-channel gates and expected-active-fraction penalty .001. Gates have a private checkpointed CPU RNG, so stochastic gating cannot change paired training examples/noise/times. Evaluation uses deterministic gates.
- `moe_dense` / `moe_topk`: four small conditional velocity experts, either all four or the top two activated by routing. Identical full architecture/weights counts; both use the same router-importance regularizer .01. This is a tiny original FM adaptation, not Time-MoE pretrained-model reproduction. All experts are evaluated with dense kernels; no acceleration claim.

Within each dense/sparse pair, the dense model's width is chosen nearest 64,000 allocated trainable parameters by a fixed integer search 4..112; the sparse model uses the identical width. Learned-threshold/gate scalars are reported as overhead. The number of potentially trainable parameters is not confused with per-example active features or physical runtime. Initialization and training data RNG are reset consistently across heads, and first-batch checksums are recorded. Comparisons with previous heads retain information-set and budget matching, but different architectures need not have the same optimization difficulty.

## Selection and evaluation

Checkpoint selection uses validation CFM loss: raw velocity MSE except whitened/balanced MLP, which use variance-weighted velocity MSE. Auxiliary reconstruction, L1, gating and routing terms regularize training but are not checkpoint-selection scores. This means the whitening intervention includes its checkpoint criterion; raw metrics are always reported for all heads.

All reported forecast and raw velocity metrics are in the same TRAIN-standardized observation coordinates, never PCA/latent coordinates. Keep PC1, PC2, mean residual PC3..F*C MSE, normalized residual diagnostic, temporal roughness (difference along time only), marginal fair CRPS, joint energy / sqrt(F*C), channel-sum CRPS / sqrt(C), ensemble mean MSE, 90% coverage and interval width. Full joint PCA residuals remain a diagnostic, not an assertion that multivariate data have rank two.

Use exactly the reference 64 evenly spaced test contexts, 32 generated futures per context and 64 Euler steps. First eight selected contexts of seed 31 receive a 128-step solver sensitivity check for every cell. Hard TopK switching may affect numerical integration; report this check. Conditional on the same history, an observed future is one draw, not the unique correct generated path.

Compare each paired sparse/dense control, each new model against reference MLP, L0 against reference U-Net, and whitened against balanced MLP. Report seed-wise effects and exploratory paired context block-bootstrap intervals (real series circular blocks 4/8; independent GP windows iid), averaging fixed seed predictions at the score level. Intervals are conditional on these seeds and this test segment; no multiplicity correction or universal method ranking claim.

## Sparse linear baseline

MultiTask Elastic Net maps flattened past lag/channel features jointly to all future outputs. Fit 12 predeclared alpha x l1-ratio candidates on TRAIN only: alpha {.001,.01,.1,1}, ratio {.5,.9,1}. Select minimum validation MSE among converged fits, no train+validation refit. Include point forecasts and whole-vector in-sample TRAIN residual bootstrap with the reference residual draw seed. This preserves empirical residual dependence but is not guaranteed calibrated. It is a forecasting baseline outside FM; neural SAE/dictionary variants remain inside FM.

## Reproducibility and limits

Save configuration, code/raw-data/reference hashes, capacities, source-specific PCA, training curves, selected checkpoints, test arrays, per-context scores, activity/reconstruction diagnostics and device. Independently check transforms, metrics and selected checkpoint replays; compare CPU/MPS numerics. Seeds are optimizer repetitions, not independent real-data periods. A single sparsity setting and fixed LR do not establish optimality. Previous test evidence motivated this follow-up, so the extension remains exploratory despite freezing the new comparison before its runs.

Primary method sources and exact adaptation boundaries: `docs/sparse_research_sources.md` and `docs/sparse_model_design.md`.

## Structural follow-up declared during the main run

Code review, before the full multivariate results were available, identified an algebraic restriction in the 64-atom velocity dictionary: all updates lie in its fixed learned span. For ETTh1 joint (F*C=168) and Exchange joint (F*C=128), this leaves at least 104/64 directions immovable. Dense and TopK controls share this restriction, but it limits comparisons with unrestricted heads. A separate frozen supplement `configs/sparse_fullrank_dictionary.json` therefore repeats BOTH dictionary variants, BOTH sources and ALL three seeds on these two datasets with 192 atoms (24 additional fits). This permits full row rank; actual rank is checked after training. Keep k=8, match dense/sparse widths within the pair and target 64k allocated parameters. The change also reallocates parameters between coefficient predictor and dictionary, so 64-versus-192-atom comparisons are not a pure isolated rank intervention. Original 264 fits are retained. This is a disclosed exploratory design correction, not a replacement selected to hide unfavorable results.

Elastic Net uses a separately fingerprinted exact centered-QR implementation (`sparse_baselines_qr.py`) to accelerate the same convex problem. Direct sklearn/QR parity and original-coordinate primal/dual certificates are checked. Its 12-candidate grid, selection and evaluation remain unchanged; no test outcome determined the acceleration.
