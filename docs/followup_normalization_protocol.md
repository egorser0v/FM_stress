# ETTh1 normalization and source-level follow-up

Frozen before scored follow-up training, 2026-10-25. This is an exploratory
follow-up of the already examined ETTh1 test period, not a new untouched test.
The original synthetic assignment and its pass rules remain unchanged.

## Questions and design

Vasya's history-normalization experiment changed level, scale, loss weighting
and the physical interpretation of the source together. We separate actionable
parts of that pipeline without claiming a unique causal explanation.

ETTh1 OT only, unchanged chronological train/validation/test intervals and
window containment. Global scaling uses training observations only. Each
window's extra transformation uses its history only, population standard
deviation, with values <=0.01 replaced by1 in train-standardized units.

Five modes: global; center only; scale only; center+scale (history);
history with loss weighted by history variance (history_rawloss). The last
control retains the history representation and source but optimizes squared
velocity errors in original train-standardized units. Validation uses the
same objective as training, so it tests the complete loss+selection policy.

All five modes cross white/GP sources and MLP/S4 heads and seeds71,72,73:
60 fits. Global and history modes additionally cross GP with its constant
level removed and GP with that level replaced by independent N(0,1/F):
24 fits. Total84. Replacement preserves each GP draw's centered shape, but
does not preserve total prior variance. Zero-level GP is deliberately singular.
Prior changes occur at training, validation and generation consistently.

All fits use MPS,6000 updates, AdamW constant lr0.001, batch128, clipping1,
weight decay0.0001 with original SSM exclusions, original width55 models.
No initialization, hyperparameter or data selection based on new test scores.
Training streams are reset after model initialization, paired across heads
and transformations; extra random levels use a separate generator.

Select minimum validation loss among steps<=2000 and steps<=6000 separately.
Keep both checkpoints, evaluate both only after all6000 updates complete.
These checkpoints are correlated budget comparisons, not extra replications.
An interrupted unfinished fit restarts fresh; completed artifacts are hash
checked on resume. Device and source hashes are recorded.

## Evaluation and interpretation

Primary fair CRPS in common train-standardized units on ALL118 test windows,
64 ensemble members, Euler64. Also save original64-context subset means,
90% coverage/width, energy score, roughness and original-unit velocity error.
This is not a bitwise replication of CUDA: new seeds and larger ensemble.
An Euler128 diagnostic uses the first8 contexts at the6000 budget for each fit.

Report per-seed results, mean contrasts and prior×head interactions. Training
seeds share a single series/test period; never call them independent datasets.
Any context bootstrap must respect serial dependence (block lengths4and8),
and is descriptive, not evidence across independent series. Center/global
tests the level-equivariant pipeline; scale/global tests scale handling;
history/center adds scale handling; history_rawloss/history isolates the
objective+checkpoint policy. Source-level controls test source geometry.
None alone proves an architecture-independent cause. Proper forecast scores
and calibration are reported separately; lower CRPS is not perfect coverage.

Software smoke runs use seed997 and only4 updates. They access this known
test dataset for implementation verification; no test score tunes the design.
All outputs live separately from earlier studies.
