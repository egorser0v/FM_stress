# Required metrics

Mean ± sample SD across optimization seeds; evaluation windows are shared.

| Cell | Seeds | PC1 velocity MSE | PCs 3–F velocity MSE | Generated roughness | Target roughness | Literal rule passes |
|---|---:|---:|---:|---:|---:|---:|
| White noise + MLP | 5 | 6.677 ± 0.084 | 0.1221 ± 0.0034 | 0.2005 ± 0.0068 | 0.1896 | 0/5 |
| White noise + S4 | 5 | 7.321 ± 0.28 | 0.1249 ± 0.0077 | 0.198 ± 0.0089 | 0.1896 | 0/5 |
| GP + MLP | 5 | 14.93 ± 0.25 | 0.04927 ± 0.00077 | 0.1931 ± 0.007 | 0.1896 | 0/5 |
| GP + S4 | 5 | 15.5 ± 0.49 | 0.05827 ± 0.0012 | 0.1969 ± 0.0084 | 0.1896 | 0/5 |
| GP + whitened MLP | 5 | 13.62 ± 0.15 | 0.0401 ± 0.00028 | 0.1926 ± 0.0039 | 0.1896 | 0/5 |

The literal rule is two-sided: residual/PC1 MSE and generated/target roughness must each lie in [0.8,1.2].
A smaller residual MSE can fail this rule. Variance-normalized diagnostics are secondary, not substituted successes.

Paired bootstrap intervals resample optimization seeds (10,000 resamples, percentile 95% CI).
They describe seed variability conditional on this fixed data draw and hyperparameters; five seeds provide limited interval resolution.
Source contrasts use source-specific velocity PCA bases and therefore also reflect different regression-label geometry.
