# Saved-model velocity-field diagnostics

These are descriptive, post hoc analyses of the 40 already completed primary
and shorter-lengthscale runs. They neither change checkpoints nor redefine the
original success rule. All errors use uncentered prediction-minus-label errors
and the original training-fitted velocity PCA. The script checks Parseval's
identity and reproduces every archived per-PC MSE before summarizing.

Run `MPLCONFIGDIR=/private/tmp/fmstress-mpl .venv/bin/python -m fm_stress.field_diagnostics`.
Outputs are in `output/field-diagnostics/`: raw JSON; per-seed PC and forecast-index
CSV; time-bin CSV; paired-difference CSV; and PNG/vector-PDF plots.

## What the existing checkpoints show

For the GP source, error in the small-variance velocity directions is much larger
for the tested S4 than for the tested MLP. This exists in **raw units**, independently
of any variance floor used in normalized diagnostics.

| Geometry | MLP mean MSE, PCs7–16 | S4 mean MSE, PCs7–16 | Whitened MLP |
|---|---:|---:|---:|
| Lengthscale 8, five seeds | 0.00000922 | 0.002806 | 0.00000439 |
| Lengthscale 6, three seeds | 0.00003887 | 0.003328 | 0.00001990 |

The S4-to-MLP ratio is about 304 at lengthscale 8 and 86 at lengthscale 6.
These large ratios concern a small absolute-error component. They must not be
described as 304-fold worse overall forecasting. PCs7–16 are a fixed descriptive
tail selected for this follow-up; they are not a newly validated success metric,
and PCA order does not by itself identify temporal frequency.

At lengthscale 8, S4 has higher mean residual-PC error in all six flow-time bins
`[0,.1), [.1,.25), [.25,.5), [.5,.75), [.75,.9), [.9,1]`.
The largest absolute residual gap is in the final bin: 0.06363 versus 0.04101
for S4 and MLP. By contrast, S4's PC1 error is slightly lower in the first two
bins. The results therefore do not support attributing the discrepancy solely
to poor history conditioning near time zero. A later-time field discrepancy is
an observation, not a diagnosed architectural cause.

The forecast-index profiles show increasing velocity error toward the far end
of the horizon for all heads. This combines conditional forecasting uncertainty
with approximation error; it does not demonstrate a boundary implementation bug.

The root agent separately checked all 15 trained GP-source primary checkpoints
on the first 128 held-out histories with identical float32 CPU/MPS inputs.
`results/diagnostics/trained_model_cpu_mps_parity.json` records a maximum absolute
prediction discrepancy of 6.68e-6 and maximum PCs7–16 CPU/MPS difference MSE of
2.74e-13. This is far below S4's observed tail error around 0.0028, so arithmetic
disagreement between these devices does not explain that discrepancy. This check
does not identify its architectural or optimization cause.

## Interpretation limits

- This compares reduced, separately trained heads under the recorded budgets.
  It does not establish an intrinsic disadvantage of S4, nor identify a causal
  effect of its state dimension, conditioning path, or optimization settings.
- Errors here are teacher-forced velocity-label errors on held-out interpolants.
  They do not directly measure generated-trajectory jitter or calibration.
- Curves show optimization-seed means and ranges. The paired percentile bootstrap
  uses 10,000 resamples of seeds, keeping source/data/evaluation pairs fixed.
  Its pointwise intervals are descriptive, not multiplicity-adjusted discovery
  claims; the three-seed follow-up is particularly limited.
- PC3 dominates much of the required PCs3–16 average. Showing individual PCs
  clarifies what that average hides without replacing the assignment metric.

The strongest next check is to compare both heads with fixed identical history
information, equal validation tuning budgets and identical per-PC loss weighting.
That experiment can test whether weighting changes the observed discrepancy.
It cannot, by itself, prove a Jacobian mechanism or universal head superiority.

## Generated sample energy in the same target basis

To connect the field analysis to actual Euler-generated samples, a separate
diagnostic projects generated and observed target patches onto **the same
training-target PCA**, subtracting the training-target mean from both. It reports
second moments around this fixed mean; these include mean bias and are neither
conditional variance estimates nor errors against paired future realizations.
There are 1,024 fixed held-out histories per geometry. No PCA axes or centering
are fitted to generated samples or test targets.

| Geometry | MLP / observed tail energy | S4 / observed tail energy | Whitened MLP / observed tail energy |
|---|---:|---:|---:|
| Lengthscale 8 | 0.253 | 338.1 | 0.930 |
| Lengthscale 6 | 0.405 | 81.4 | 0.907 |

Here tail energy is the mean second moment in target PCs7–16, the same descriptive
index subset for every head. Observed tail energies are only 7.14e-6 and 3.17e-5
respectively. Thus very large ratios still concern a small absolute component.
For these checkpoints, S4 produces excess energy in those small-variance target
directions; raw MLP suppresses it; whitened MLP comes closer to the observed level.
This is compatible with all heads passing average adjacent-jump roughness.
It does not show full conditional calibration, nor prove that matching this tail
alone gives a better distribution. Keep the complete PC plots visible rather
than presenting only the favorable tail summary.

The figures `sample_target_pc_energy.png` and `.pdf` show every target PC with
seed means/ranges. Per-seed values are in `sample_target_pc_second_moments.csv`
and the diagnostic JSON. This is a post hoc supplementary finding; the original
primary metrics, checkpoint selection and raw success flags remain unchanged.
