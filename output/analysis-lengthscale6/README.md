# Analysis provenance

Results directory: `results/lengthscale6`
Complete: **True**
Observed/expected runs: 15/15

- Summary error bars are sample SD across optimization seeds, not standard errors.
- Paired 95% bootstrap intervals resample matched seeds, 10,000 replicates, fixed seed 917.
- Paths use seed 0 and the first six test contexts, never visually selected examples.
- Solid and dashed paths are independent stochastic draws conditional on the same history; pointwise discrepancy is not a forecast metric.
- The GP oracle has access to raw history statistics unavailable to the learned heads.
- Learning-curve objective differs for the whitened control; raw validation MSE is shown separately.
- Per-PC axes are fitted to training velocity draws separately for each source.
- PNG and vector PDF versions of every figure are exported.
