# Learned velocity templates: size supplement

Each dictionary element is a learned velocity template, also called a dictionary
atom: the head predicts coefficients that combine these templates into its
velocity output. Dense heads can use every template; TopK keeps eight coefficients.
The machine-readable results retain the conventional `atoms` field name.

This exploratory follow-up was declared after inspecting the main sparse suite.
It addresses a limitation of the original velocity head: its 64-atom decoder can
span at most 64 directions, whereas the ETTh1 seven-channel future has 168 entries
and the Exchange eight-channel future has 128. A dense coefficient vector does
not remove that decoder rank bound. This is a limit on the velocity field, not
an assertion that generated paths have rank 64: source variation outside the
dictionary span can remain in the generated endpoint.

The supplement adds **24 fits**: two datasets, dense and TopK dictionary heads,
white and GP sources, and seeds 31/32/33. It uses 192 atoms, which removes the
atom-count cap below the output dimension. TopK remains `k=8`. Every comparison
uses the 24 corresponding original 64-atom fits; none is selected by test outcome.
The independent audit must check the actual learned decoder rank instead of
equating sufficient atom count with learned full rank.

All dataset splits, TRAIN-only normalization, source covariance, velocity PCA,
2,000 training steps, optimizer settings, validation checkpoint selection,
64-context/32-sample forecasts, Euler budget and evaluation seeds are unchanged.
The analyzer verifies manifests, source and dataset hashes, checkpoint/sample
hashes, evaluation indices, and first/last training random-batch hashes.

## Comparisons and limits

- **192 minus 64**, separately for dense and TopK heads under each source.
- **TopK minus dense**, separately at 64 and 192 atoms under each source.
- **Interaction:** `(TopK192 - dense192) - (TopK64 - dense64)`.

Report fair marginal CRPS, scaled joint energy, roughness ratio, PC1 velocity MSE,
residual velocity MSE and normalized residual velocity MSE. Seed comparisons use
the mean and sample SD of three paired differences. Proper-score comparisons also
use paired circular moving-block bootstrap intervals with block lengths four and
eight, 10,000 repetitions, and fixed bootstrap seed 9141. Seed-averaged ordered
context differences are formed before resampling. These intervals are conditional
on the fitted models and fixed test period, and are not multiplicity-adjusted.

The parameter target remains about 64,000, so a larger decoder leaves fewer
parameters for its coefficient predictor and changes its width. Therefore
192-versus-64 is **not a clean rank-only causal intervention**. Within each size,
the dense and TopK pair does retain identical stored capacity and predictor width.
Keeping `k=8` also reduces the active fraction from 8/64 to 8/192 while expanding
the available dictionary. Report the dimensions, actual widths and parameter
counts alongside outcomes.
Even a full-rank learned decoder does not imply that eight active atoms can
represent every target velocity; finite-sample atom usage is a separate diagnostic.

| Dataset | Future dimension | 64 atoms: width / parameters | 192 atoms: width / parameters |
|---|---:|---:|---:|
| ETTh1, seven channels | 168 | 40 / 64,696 | 26 / 64,246 |
| Exchange, eight channels | 128 | 43 / 63,339 | 32 / 63,520 |

Each row uses the same listed capacity for its dense and TopK heads. These are
stored trainable parameter counts, not the number of nonzero coefficients per
example or a measured compute-saving claim.

## Reproduce the analysis

After both sets contain all 24 required fits:

```sh
.venv/bin/python -m fm_stress.analyze_dictionary_supplement
```

The command fails before producing a comparative summary when either set is
incomplete or provenance differs. Outputs in `output/dictionary-supplement` are
`summary.json`, `aggregate.csv`, `paired_seed_contrasts.csv`,
`paired_context_contrasts.csv`, and `dictionary_scores.png/.pdf`.
Use `--rank-diagnostics results/diagnostics/dictionary_span.json` together with
`--rank-diagnostics results/diagnostics/dictionary_span_fullrank.json` to attach
the two complete independent learned-rank audits. Their fingerprints, checkpoint
hashes and sample hashes must cover exactly all 48 compared fitted models; both
audit file checksums are retained. Absence of that optional attachment is
explicitly recorded and must not be interpreted as confirmation of learned rank.
