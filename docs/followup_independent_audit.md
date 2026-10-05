# Independent audit of the follow-up design

Reviewer: independent science-audit subagent. This note records checks of the
design and numerical code before the new results were available. It does not
claim that unfinished runs passed or that either head must win.

## Existing experiment

Read both supplied project PDFs, all four original numerical modules, model
provenance, original protocol and saved final conclusions. Reran the delivery
verifier: all 40 base runs and nine optimization restarts passed array-to-metric
recomputation and the saved checkpoint-selection/provenance checks. Independent
CPU test execution gave 25 passes and one skipped MPS test. Root separately ran
trained-checkpoint CPU/MPS parity; that device check is not attributed to this
CPU-only reviewer. No blocking mathematical implementation error was found.

Residual limitations are documented: original learned encoders diverge after
identical initialization; reduced S4 state size and common learning rate do not
establish optimal tuning; whitening also weights the objective; the raw equality
criterion rejects low residual error; and roughness alone does not establish
conditional calibration. None licenses changing the original primary outcomes.

## Controlled head/objective experiment: prelaunch review passed

Read `controlled_followup.py` and `controlled_followup_protocol.md`.

- `Identity` history encoders give both heads identical complete normalized
  histories, with no separately trained representation.
- The balanced loss computes squared projected errors divided by training
  velocity variances with the declared floor. Inputs and outputs remain in
  temporal observation coordinates, so S4 never convolves across PCA indices.
- Training examples, source draws and times are paired by resetting RNG state
  after architecture-dependent construction.
- Preparation generates only train/validation data. All eight equal-budget
  learning-rate pilots must finish before a hashed selection file is locked.
  Each head/objective selects using its own validation objective, not test error.
- Test data are first generated for final evaluation. Every final cell uses
  the locked learning rate, same update budget and same Euler steps. Source
  hashes prevent mixing code revisions in the same result directory.

This is a bounded two-rate, one-pilot-seed search, not proof of optimal tuning.
Three final seeds and one fresh GP draw support a limited controlled comparison.
Changing directional loss weighting can change gradient clipping and optimization
as well as error priorities; a favorable outcome cannot by itself identify a
Jacobian mechanism. Identity conditioning does not eliminate all positional or
architectural differences. New five-bin diagnostics and archived six-bin
diagnostics have different boundaries and must not be directly conflated.
Each cell selects its own learning rate; if rates differ, raw-versus-balanced
comparisons concern equally tuned training pipelines, not an isolated change of
loss weighting. Relative to the old learned-encoder runs, fresh data, seeds,
training budget and conditioning also change. Cross-experiment differences
cannot identify the effect of replacing the encoder alone.

## Conditional-ensemble experiment: prelaunch review passed

Read `ensemble_evaluation.py` and `ensemble_evaluation_protocol.md`.

- All checkpoints use the same fresh histories and independently drawn source
  ensembles; repeated histories align with flattened source samples.
- No training or checkpoint selection uses the fresh evaluation draw.
- Independently checked fair CRPS and energy formulas against explicit full
  pairwise arrays on random data. Also checked their equality for one-dimensional
  paths. Off-diagonal denominators are correct; horizon dimensions are averaged
  within a context, not treated as independent samples.
- The bootstrap first averages scores over fixed model seeds, then resamples
  paired histories. Its intervals therefore describe fresh-history uncertainty
  conditional on these models and source ensembles, not training-data uncertainty.
- Marginal coverage uses declared empirical quantiles; width and variance are
  not interpreted as rewards for narrow distributions. The exact GP reference
  correctly discloses its privileged raw-history information.
- Completion requires all 15 specified checkpoints and consistent fingerprints;
  subset runs are marked partial. Original files are preserved.

The proper-score comparisons are exploratory and not multiplicity-adjusted.
One realized future per context is valid for average proper scores but gives
limited power at 256 histories. Coverage is marginal, not simultaneous path
coverage. A method can improve energy score yet retain errors in small-variance
directions; retain the full PC plots alongside global scores.

## Pending final review

After runs finish, independently verify complete run counts, locked pilot
selection, sampled metric recomputation from saved arrays, manifests and final
interpretation. Prelaunch approval is not a substitute for those outcome checks.

### Analysis-code review and completion gate

Reviewed `analyze_controlled.py` and `analyze_ensemble.py`. Controlled seed
pairing, mean/SD aggregation, head-by-loss differences and paired percentile
bootstrap formulas are correct. The ensemble runner correctly forms paired
history differences after averaging scores across fixed trained seeds. Neither
analysis treats future coordinates or ensemble members as independent histories.

The presentation analyzers alone are not exhaustive verifiers. In particular,
the ensemble analyzer checks sample shapes and fingerprint labels but trusts
saved per-context metrics and summaries. The controlled analyzer recomputes core
metrics but does not independently reconstruct the pilot argmin or regenerated
test data. These are verification gaps, not observed result errors.

`scripts/check_followup.py` supplies the independent CPU-only completion gate:

1. Check actual source/checkpoint file hashes and exact configured run sets.
2. Recompute all eight pilot minima, tie-breaking and locked learning-rate
   choices; verify final checkpoints correspond to the first validation argmin.
3. Regenerate the training PCA and fresh controlled test/source/time arrays,
   recompute all archived metrics, and compare CPU checkpoint predictions on
   64 fixed inputs with saved MPS predictions.
4. Independently recompute controlled means, SDs and paired bootstrap intervals.
5. Regenerate common ensemble inputs and the exact privileged-reference draws;
   recompute every learned per-context metric, head summary and paired-history
   bootstrap interval from saved arrays.
6. Recompute all three solver sensitivity comparisons from their saved Euler-128
   and Euler-256 ensembles, including endpoint RMSE and contextwise score changes.

Only a successful complete invocation writes
`results/diagnostics/followup_verification.json` with `scope="both"`. A
single-suite invocation records its narrower scope and must not be described
as verification of both suites. The final report's numerical claims still
require comparison with these verified outputs.

## Completed controlled-suite verification

The independent CPU verifier passed all eight pilots and twelve final models.
The evidence is `results/diagnostics/controlled_verification.json` with
`scope="controlled"`. It verified actual source hashes, the true learning-rate
argmin (all four cells select 0.001), first-minimum validation checkpoints,
regenerated train PCA and fresh test/source/time arrays, all raw/normalized
metrics, target-PCA sample moments, aggregates and paired bootstrap intervals.
Checkpoint predictions recomputed on CPU differ from the stored MPS predictions
by at most 5.60e-6 on the audited 64 inputs. Training device evidence is the MPS
run records; this verifier itself uses CPU and does not count as new GPU training.

The findings are a tradeoff, not a repaired universal S4 advantage. S4's balanced
loss reduces its normalized residual error from 31.1219 to 0.15766, below balanced
MLP's 0.17458. Its target-PC tail ratio falls from 597.2 to 0.585; balanced MLP is
0.606, and the paired difference in those tail ratios has an interval spanning
zero. However, S4 raw residual MSE increases on average from 0.05706 to 0.06059,
and PC1 MSE from 14.2555 to 22.7211. MLP's PC1 error deteriorates substantially
under balanced temporal-coordinate loss, from 14.0029 to 85.2370. A conclusion
that balancing improves every metric, that S4 wins overall, or that changing
only the loss reproduces the earlier fully whitened control would be incorrect.

Both loss types selected the same nominal learning rate in this realized suite,
but gradient weighting and clipping dynamics still differ. These experiments
support a metric-dependent effect of the objective under this finite budget.
They do not prove a specific Jacobian cause or optimizer-independent ranking.

## Completed ensemble and joint verification

The complete CPU audit now passes with `scope="both"` in
`results/diagnostics/followup_verification.json`. In addition to the controlled
checks above, it verified all 15 MPS ensemble evaluations and three Euler-256
sensitivity evaluations. It regenerated fresh histories/source ensembles exactly,
recomputed every per-context metric, model mean/SD and paired-history bootstrap
interval, checked actual source/checkpoint hashes, regenerated privileged exact
conditional draws and recomputed solver endpoint/score differences from arrays.
This independent analysis did not train or generate learned ensembles on GPU;
those actual MPS operations are evidenced separately by their saved run records.

On the new 256-history evaluation, mean fair CRPS is 0.466306 for MLP, 0.461315
for S4 and 0.460568 for whitened MLP. Mean fair energy scores are 2.413208,
2.392024 and 2.386600 respectively. Both alternatives have slightly better point
estimates than MLP, but all four declared proper-score difference intervals
include zero. For example, S4-minus-MLP CRPS is -0.004991 with interval
[-0.016784, 0.006417]; its energy difference is -0.021184 with interval
[-0.076241, 0.032345]. This is insufficient evidence of a proper-score winner,
and is not an equivalence result. Marginal 90% coverage is 91.13%, 90.31% and
90.63%; these point estimates must not be portrayed as proof of full calibration.

On the prescribed first-32-history seed-0 subset, doubling Euler steps changes
CRPS means by at most 0.000155 and energy means by at most 0.001042. Paired
endpoint RMS differences are approximately 0.0127–0.0137. This supports limited
numerical sensitivity for these scores on this subset; it does not establish
solver convergence in every conditional covariance direction.

The additional science therefore concerns **objective-dependent directional
behavior**, with tentative rather than decisive forecast-score differences.
Generated tail-energy and proper-score results answer different questions and
are not contradictory. None overrides the original assignment's reported flags.

### Report prepublication correction

Independent review found a source-filter omission in the draft follow-up report
builder: its archived target-PC energy table initially pooled white and GP cells
for MLP/S4, while the figure and text described GP only. The root agent corrected
the filter to `source == "gp"` before report generation. The experiment arrays
and analysis artifacts were unaffected. Final PDF text/figure verification is
recorded separately after export.
