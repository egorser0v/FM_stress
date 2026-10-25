# Full-ensemble Euler sensitivity for conditional-mean supervision

This is a post-hoc numerical check of the completed 24-fit label study. No checkpoint, hyperparameter, label or training input is changed. The original Euler-64 results remain the primary recorded experiment.

For every saved checkpoint, reuse all 256 forecast histories, all 32 source draws per history and the same normalized targets. Reconstruct the initial source draws from their original seed and require exact agreement with the saved arrays. On MPS, first replay all 64-step paths and require relative RMS agreement below 1e-5 with the original ensemble. Then generate the full 128-step ensemble with the same batch size of 1024. Freeze this script, configuration, original checkpoint/sample hashes and prerequisite audit hashes before generating new outcomes. There is no CPU fallback for this generation run.

Retain per-context CRPS, energy score, interval coverage and width, forecast mean error, ensemble variance and roughness at both resolutions. Report paired conditional-mean-minus-ordinary-label contrasts for every data/initialization replicate, their changes and any sign reversals. A sign reversal in a tiny difference is a numerical sensitivity finding, not evidence of a meaningful effect in the opposite direction. No new pass threshold or significance test is introduced.

Verification includes direct pairwise CRPS and independent order-statistic coverage arithmetic on all ensembles. A separate CPU replay regenerates 128-step paths for the first eight histories of every model, with a relative RMS tolerance of 2e-4. Check all frozen input hashes again after execution. Store new arrays and audit records in results/followup-label-solver, and a compact report in output/followup-label-solver.

The planned workload is 24 checkpoints, 8192 paths per checkpoint, 64 plus 128 Euler updates. Timing is measured for each resolution and model. This compares two finite numerical resolutions; it cannot prove exact ODE integration or create additional independent training replicates.
