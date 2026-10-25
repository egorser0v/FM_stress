# No-flow negative control

This checkpoint-free baseline sets the velocity field to zero. Every forecast equals its initial source and ignores the observed history. It is a diagnostic of source-only apparent success, not a competing learned model. All 12 frozen capacity kernel/source/draw groups are scored; no result selects settings or data. The third test draw is accessed explicitly as part of this fixed scoring plan.

Metrics below average three independent draws. Raw velocity metrics use training-only, source-specific velocity PCA; forecast metrics use identical 256 contexts and 32 source draws per context. Compare raw velocity errors within a source, since source noise floors differ.

| Target / source | PC1 MSE | Residual MSE | Roughness ratio | Fair CRPS | 90% coverage | Literal successes |
|---|---:|---:|---:|---:|---:|---:|
| rbf / white | 40.339637 | 1.692008 | 9.6012 | 0.734303 | 0.7579 | 0/3 |
| rbf / gp | 58.196578 | 1.161722 | 0.8459 | 0.733679 | 0.7570 | 0/3 |
| ou / white | 27.017454 | 1.775407 | 2.6291 | 0.673176 | 0.7865 | 0/3 |
| ou / gp | 40.722329 | 1.392967 | 0.9013 | 0.673113 | 0.7879 | 0/3 |

Direct O(S²) pairwise fair CRPS independently agrees with the sorted-order formula to a maximum absolute difference of 8.88e-16. All individual results, input hashes, source covariance/PCA hashes, generated arrays and per-context proper/calibration scores are retained under `results/followup-capacity-baselines/`. Original capacity results and fit counts are unchanged.
