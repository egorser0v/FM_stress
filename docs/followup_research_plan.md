# Follow-up after Lyamin's 0333803

Study date: 2026-10-25. Starting revision:0333803e2563b38b98073b4ed2009c41e3889bfd.
No new commits or pushes are authorized for this follow-up. The working copy was
fast-forwarded to the existing colleague commit before adding separate study files.
Original reports/results are preserved.

## Scope

1. Independently reconstruct the360 saved kernel-scan velocity results, replay
   the18 most relevant complete forecast cells, verify hashes, and quantify the
   invariant output-nullspace of the width56/horizon64 MLP.
2. Derive and validate the conditional field given normalized history (the same
   information available to the network), then repeat the25 original-checkpoint
   Jacobian comparisons. Separate equal-direction and covariance-weighted local
   derivative errors. The latter is a post-hoc diagnostic, not a forecast score.
3. Capacity/source/budget study:36 fresh MPS trainings, separately selected best
   checkpoints at6000 and12000 updates. RBF and OU, F64, white/GP, independent
   draws and paired seeds. Test decoder capacity at near-equal parameter count
   and with a wider same-depth MLP. See followup_capacity_protocol.md.
4. ETTh1 normalization/source-level study:84 fresh MPS trainings, checkpoints
   selected at2000 and6000 updates. Center/scale factorial, raw-unit loss control,
   zero/replaced GP level controls. See followup_normalization_protocol.md.
5. Conditional-label-noise intervention:24 fresh MPS trainings,6000 updates.
   Identical stochastic input streams but either usual noisy CFM labels or the
   information-matched conditional mean. White/GP × MLP/S4 ×3fresh draws.
   This was added after the initial oracle analysis to test the proposed
   mechanism directly. It is an oracle-assisted diagnostic, not a deployable
   forecasting method. See followup_label_noise_protocol.md.

Total planned new scored trainings:144. Multiple checkpoints from one training
are not independent replications. Software smokes are excluded from counts.

An additional post-hoc numerical check regenerates the complete forecasting
ensembles of all 36 final-budget capacity models at 128 Euler steps, paired
with their saved 64-step ensembles. This evaluates integration sensitivity of
small forecasting-score differences; it does not retrain or select models.
It was added after observing the first two draws' small CRPS head differences.
Every planned cell is included, not only cells with a preferred ranking.
The original 64-step results remain the primary scores.

A further post-hoc oracle check applies finite covariance-scaled residual
perturbations to all 25 original models: 256 histories, four shared directions,
five flow times and two fixed scales. It checks whether the weighted local
derivative ranking also appears in actual nearby velocity changes. Absolute
errors and reference denominators are retained. This is a geometry-sensitive
diagnostic, not a causal optimization test or a forecast score.

A no-flow baseline scores all 12 capacity geometry/source/draw groups with
zero velocity and unchanged source samples. It checks how much apparent
smoothness is supplied by the prior alone. Test inputs are scored after the
capacity design is frozen, including the still-running third draw; neither
hyperparameters nor checkpoint selection use this baseline. Independent
pairwise and sorted CRPS formulas agree. No training runs are added.

After aggregate center-only and history/raw-loss results showed similar CRPS
but different interval coverage, an exploratory calibration analysis was added.
Test windows are grouped by observed-history standard deviation using training
quartiles (below Q25, Q25--Q75, above Q75). These cutpoints and reported metrics
are fixed before viewing stratified outcomes. All completed pipelines and ridge
baselines are included, with weighted-bin reconstruction checks. This analysis
does not tune models or change the primary scores.

After the label study completed, its numerical check was expanded from eight
contexts of the first draw to all 24 models and full 256-context, 32-member
ensembles. A frozen post-hoc MPS run replays every Euler64 forecast and generates
Euler128 forecasts with the same initial states and checkpoints. CPU128 replay
on eight contexts per model and a separate-agent all-array CRPS/hash/pairing
audit verify the result. No additional training or model selection is involved.

## Evidence discipline

Per-package configurations and numerical source hashes are frozen before scored
training. This is local documentation and logs, not externally preregistered
version history. Previous test results informed the follow-up questions.
New synthetic draws are independent of earlier draws. ETTh1's existing test
period has been examined previously and cannot be called untouched.

No causal claim follows from a Jacobian norm alone. A privileged Gaussian
oracle and an information-matched nonlinear oracle must remain distinct.
The static decoder nullspace is an exact architecture constraint; the observed
effect of widening additionally changes parameterization/optimization.
History normalization changes the complete source/data/loss pipeline.
Report all seeds, all planned cells, calibration and solver diagnostics;
do not suppress failures or choose a method using test scores.

All new outputs are under results/followup-* and output/followup-*.
The final report must distinguish independently reproduced old numbers,
new training, exploratory diagnostics and remaining uncertainty.
