# Required metrics

Mean ± sample SD across optimization seeds; evaluation windows are shared.

| Cell | Seeds | PC1 velocity MSE | PCs 3–F velocity MSE | Generated roughness | Target roughness | Literal rule passes |
|---|---:|---:|---:|---:|---:|---:|
| White noise + MLP | 3 | 6.591 ± 0.13 | 0.1902 ± 0.0021 | 0.2223 ± 0.0039 | 0.2257 | 0/3 |
| White noise + S4 | 3 | 6.561 ± 0.11 | 0.1883 ± 0.0018 | 0.2162 ± 0.0026 | 0.2257 | 0/3 |
| GP + MLP | 3 | 17.47 ± 0.21 | 0.1237 ± 0.002 | 0.2124 ± 0.0065 | 0.2257 | 0/3 |
| GP + S4 | 3 | 17.26 ± 0.023 | 0.1311 ± 0.00067 | 0.2108 ± 0.0088 | 0.2257 | 0/3 |
| GP + whitened MLP | 3 | 16.54 ± 0.21 | 0.1086 ± 0.0018 | 0.2129 ± 0.0058 | 0.2257 | 0/3 |

The literal rule is two-sided: residual/PC1 MSE and generated/target roughness must each lie in [0.8,1.2].
A smaller residual MSE can fail this rule. Variance-normalized diagnostics are secondary, not substituted successes.

Paired bootstrap intervals resample optimization seeds (10,000 resamples, percentile 95% CI).
They describe seed variability conditional on this fixed data draw and hyperparameters; five seeds provide limited interval resolution.
Source contrasts use source-specific velocity PCA bases and therefore also reflect different regression-label geometry.
