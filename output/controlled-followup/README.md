# Controlled follow-up: all planned runs completed

Eight validation-only learning-rate pilots and twelve final MPS runs; all saved final metrics were recomputed from sample arrays. No previous results are replaced.

| Head | Objective | PC1 MSE | Residual MSE | Normalized residual MSE | Roughness ratio | Path PC7–16 moment ratio |
|---|---|---:|---:|---:|---:|---:|
| MLP | raw | 14.00289 | 0.046687 | 0.22222 | 1.0098 | 0.267 |
| MLP | balanced | 85.23700 | 0.043891 | 0.17458 | 0.8659 | 0.606 |
| S4 | raw | 14.25550 | 0.057060 | 31.12191 | 1.0967 | 597.199 |
| S4 | balanced | 22.72105 | 0.060587 | 0.15766 | 0.9471 | 0.585 |

Both networks receive identical normalized history through an identity encoder. Balanced loss changes error weighting only; the S4 sequence remains indexed by time. Rate selection uses separate validation objectives and equal pilot budgets.

Paired bootstrap of three optimization seeds conditional on one fresh GP dataset; exploratory, not population evidence.

Full paired contrasts, interactions, selected rates and per-run values are supplied in summary.json and runs.csv.
