# Full-ensemble numerical integration check

Prepared before the new integration evaluation, 25 October 2026. This is a post-hoc numerical diagnostic motivated by small forecasting-score contrasts. It does not introduce training, select models, or replace the recorded capacity results.

The fixed inclusion set is **all 36 capacity models** at their independently selected final-budget (12,000-update prefix) checkpoint. Selection remains the original validation minimum, not the last training step or a forecast score. Every model receives all 256 original test contexts and all 32 original source draws per context. Euler integration is repeated at 128 steps; the original 64-step ensemble remains the comparison. There is no outcome-based choice of kernels, sources, heads or draws.

Independent NumPy code reconstructs raw test windows using data seed + 2, history-only normalization, and the complete ensemble source using data seed + 1000. All test-input hashes, checkpoint hashes and original ensemble hashes must match the frozen capacity records. The architecture is reused to load weights. The first two complete ensembles are replayed with Euler64 on the chosen inference device; relative RMS must be at most 2e-4 before refinement proceeds. The same 256-path batching and float32 representation are used as in the original generation. Euler128 then uses every original context and source draw.

Independent scoring recomputes the original 64-step per-context metrics and checks them against the saved results. It computes fair CRPS, fair energy score, ensemble-mean MSE, 80% and 90% coverage/width at both resolutions, as well as endpoint RMS change and ensemble roughness. Saved arrays include both complete ensembles, normalized histories, original source draws, targets and paired per-context scores. Head-minus-S4 CRPS contrasts are paired by kernel, source and data draw; all individual draws and changes of mean ordering are reported. Negative differences favor the named head. Models are neither selected nor ranked by a threshold introduced after looking at refinement outcomes.

Configurations and numerical source hashes are frozen in `results/followup-solver/manifest.json`. CPU smoke is fixed to two contexts for RBF/white/original-MLP and RBF/white/S4 at data seed 607041; it verifies implementation and estimates runtime. Full scored generation uses MPS only once the coordinator grants a slot. The two-context CPU smoke is excluded from the all-model result count and does not adjust experiment settings. Resumption checks immutable inputs and saved output hashes.

A change between 64 and 128 steps demonstrates discretization sensitivity. Stability at those resolutions does not prove convergence to the exact flow, and a tiny head ordering should not be presented as robust merely because its mean sign survives. Ensemble Monte Carlo and finite-context uncertainty remain. Original reports, capacity records, fit counts and success flags stay unchanged. No commits or pushes.

```sh
.venv/bin/python scripts/audit_followup_solver.py --prepare
.venv/bin/python scripts/audit_followup_solver.py --smoke --device cpu
# Only after an MPS slot is granted:
.venv/bin/python scripts/audit_followup_solver.py --device mps
```

Implementation review before full evaluation added explicit checkpoint/run identity checks and documented-protocol archival support. Initial two-context CPU smokes are retained under `results/followup-solver/pre-review-smoke`; the finalized code is frozen again and its smokes rerun before MPS evaluation. Numerical settings are unchanged. If the public capacity protocol is clarified after training, its original bytes must still match the original manifest at `results/followup-capacity/frozen_protocol.md`; numerical source files receive no such alternate-path exception.
