# Sparse/whitening extension: analysis fixed before completion

The analysis combines 264 newly trained flow-matching models with 96 frozen
controls from the preceding extension. New model families, seeds, sources,
datasets, checkpoint selection and training budget are fixed in
`configs/sparse_extension.json`. Elastic Net is an additional forecasting
baseline, not a flow-matching component. Existing control outcomes motivated
this exploratory extension; it is not a prospectively independent replication.

## Comparisons and interpretation

Every new head is compared with the same-source dense MLP. Additional comparisons
pair TopK history AE with dense history AE, sparse convolutional history with its
dense iterative control, TopK velocity dictionary with dense dictionary, and
Top-2 routing with dense experts. Each pair shares its width. Convolutional
sparsity additionally introduces learned thresholds and an L1 objective.
The gated U-Net is compared with its original same-width U-Net. These are
architectural/regularization interventions, not equal claims about parameter
pruning, activation sparsity or actual computation speed.

Whitened MLP is compared both with ordinary MLP and with variance-balanced MLP.
The latter retains temporal observation inputs and changes the objective only;
full-rank PCA whitening additionally changes the input/output coordinates. All
forecast scores and required raw velocity errors are measured after restoring
observation coordinates. PCA is fit on TRAIN velocity samples separately for
each source, with the existing declared eigenvalue floor and no truncation.

For every head, GP versus temporally white source is compared on common target
forecasts. Source/head interactions are `(head GP − MLP GP) − (head white − MLP
white)`. Source changes velocity labels and PCA axes, so the analyzer does not
interpret cross-source differences in raw PC velocity errors as source wins.

## Scores and uncertainty

Tables retain PC1 MSE, residual PC3…F*C MSE, normalized residual MSE, roughness
ratio, fair marginal CRPS and scaled joint energy. All saved scalar diagnostics
are available in CSV. Aggregates report the three individual seeds through raw
records and mean/SD summaries. Paired contrasts retain every seed difference,
its mean, SD and range. Roughness ratio is a calibration diagnostic with target
one: a negative difference is not automatically an improvement.

Proper-score contrasts additionally use the preceding analyzer's fixed 10,000
replicate percentile bootstrap: iid selected contexts for synthetic independent
windows; circular moving blocks of four and eight ordered selected origins for
real datasets. Context differences are paired and averaged over the three fitted
seeds before resampling. These intervals condition on fitted models and this
test period, exclude training-dataset/model-selection uncertainty, and are not
adjusted for the many exploratory comparisons. Three-seed bootstrap intervals
are descriptive and are not evidence of population-level significance.

## Diagnostics and artifact integrity

Measured code activity, deterministic L0 gate activity and expert routing
activity are reported separately. A regularization name alone does not establish
that a network actually pruned anything. Checkpoint selection at the last update
is disclosed as a budget-sensitivity diagnostic, not a convergence test. Solver
comparisons preserve the declared first-eight-context doubled-Euler subset.

The analyzer verifies run identities, hashes, checkpoint selection, context
indices and recorded means. Independent CPU replay and direct score equations
belong to `scripts/check_sparse_extension.py`, not to the analyzer's validation
claim. Strict analysis requires all 264 new fits and all four Elastic Net
baselines. `--allow-partial` marks output incomplete and suppresses comparative
intervals; it is for development only. Existing controls and scientific source
modules are never overwritten. Figures use the same teal/blue and orange source
colors, shared model ordering, seed-SD error bars and explicit baseline lines.

## Dictionary span and structural supplement

The original dictionary head predicts `v = D a`, with one fixed learned matrix
`D` and 64 atoms. If `P` projects onto the orthogonal complement of `span(D)`,
then `P v = 0` and every trajectory satisfies `P x(1) = P epsilon`. TopK with
eight active coefficients may use different atoms at different times, so its
trajectory is not restricted to eight fixed atoms; however, the union remains
inside the global 64-atom span. Dense and TopK share this architectural limit.

The joint future has 48 coordinates for GP3 and 24 for ETTh1-OT, where 64 atoms
can span the full space. ETTh1-7 has 168 coordinates and Exchange has 128, leaving
at least 104 and 64 orthogonal directions unchanged by the original head.
Consequently, comparison with unrestricted MLP also changes the field's rank
capacity. A weak result from this particular head is not general evidence
against sparse velocity representations.

`scripts/check_dictionary_span.py` verifies checkpoint/sample/source/data hashes,
computes singular values and numerical rank, reconstructs the identical initial
source draws and measures endpoint invariance directly from stored forecasts.
It also checks that predicted velocities lie inside the learned span. An exact
component of fixed-test velocity MSE is the squared target velocity projected
onto the orthogonal complement. For a fresh finite ensemble of size S, the
expected orthogonal ensemble-mean forecast error is the selected target's
orthogonal squared norm plus the projected source covariance trace divided by S,
with both normalized by joint dimension. This expectation is not a deterministic
bound on a particular stored random ensemble. No CRPS/energy lower bound is
claimed. Validation atom usage is recorded separately from matrix rank; atoms
unused on these validation interpolants are not proven dead on every input.

A declared structural supplement repeats only the two dictionary heads on
ETTh1-7 and Exchange with 192 atoms, both sources and all three seeds (24 fits).
It retains the original 264 fits, uses the same frozen numerical implementation,
and selects widths at the same approximate allocated-parameter budget. The
larger dictionary can span all 168/128 coordinates, while TopK remains eight.
This is a rank-capacity sensitivity study: changing the dictionary size also
changes width/capacity allocation, and both must be reported. Even a full-rank
dictionary may learn to use only a small subset of atoms. The supplement was
declared after identifying the architectural constraint; it is an exploratory
follow-up rather than an independent preregistered test.

Main and supplementary span diagnostics are saved separately as
`results/diagnostics/dictionary_span.json` and
`results/diagnostics/dictionary_span_fullrank.json`. They never replace original
results or alter a trained checkpoint.
