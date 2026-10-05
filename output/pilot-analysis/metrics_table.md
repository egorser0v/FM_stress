# Required metrics

Mean ± sample SD across optimization seeds; evaluation windows are shared.

| Cell | Seeds | PC1 velocity MSE | PCs 3–F velocity MSE | Generated roughness | Target roughness | Literal rule passes |
|---|---:|---:|---:|---:|---:|---:|
| White noise + MLP | 1 | 9.281 (one seed) | 0.3371 (one seed) | 0.2812 (one seed) | 0.1911 (one seed) | 0/1 |
| White noise + S4 | 1 | 10.86 (one seed) | 0.4447 (one seed) | 0.3444 (one seed) | 0.1911 (one seed) | 0/1 |
| GP + MLP | 1 | 26.12 (one seed) | 0.09089 (one seed) | 0.1916 (one seed) | 0.1911 (one seed) | 0/1 |
| GP + S4 | 1 | 24.74 (one seed) | 0.1405 (one seed) | 0.1893 (one seed) | 0.1911 (one seed) | 0/1 |
| GP + whitened MLP | 1 | 24.65 (one seed) | 0.05877 (one seed) | 0.1936 (one seed) | 0.1911 (one seed) | 0/1 |

The literal rule is two-sided: residual/PC1 MSE and generated/target roughness must each lie in [0.8,1.2].
A smaller residual MSE can fail this rule. Variance-normalized diagnostics are secondary, not substituted successes.

Paired bootstrap intervals resample optimization seeds (10,000 resamples, percentile 95% CI).
They describe seed variability conditional on this fixed data draw and hyperparameters; five seeds provide limited interval resolution.
Source contrasts use source-specific velocity PCA bases and therefore also reflect different regression-label geometry.
