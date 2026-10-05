# Limited optimization-budget sensitivity

The primary experiment uses 6,000 updates per cell with constant learning rate
0.001 and validation-selected checkpoints. An interim independent audit of
the first completed seeds found continued improvement in both heads, rather
than clear evidence that S4 alone was cut off before learning. Between 4,500
and 6,000 steps the best validation objective improved by up to 8.8% for
white-source S4 and by up to 4.6% for GP-source MLP. Several GP-S4 final
checkpoints were 8–9% worse than their selected best checkpoint, consistent
with late learning-rate noise. These observations do not establish convergence.

All inspected S4 kernels were internally stable. The optimizer applies weight
decay 0.0001 to 57,049 ordinary parameters and zero decay to 8,232 SSM
transition/input/step parameters. A direct check confirms these groups survive
the transfer to MPS. Internal linear SSM stability does not prove global
stability of the nonlinear flow ODE.

## Supplemental protocol, fixed before running it

For **GP-source MLP, S4 and whitened MLP**, initially seed 0:

1. Restore each validation-selected primary checkpoint.
2. Reset Adam and train for **3,000 additional updates at learning rate 0.0003**.
   Keep the dataset, batch size, clipping, weight decay rules, encoder,
   architecture and loss unchanged. Reset the training random stream identically
   across heads to `30000 + seed`.
3. Check validation every 500 updates. Select from these checkpoints **and the
   original checkpoint**, using each head's original training objective.
4. Evaluate the selected model on the original fixed test pairs, with the same
   64 Euler steps and path count. Save this in a separate directory; primary
   records are never overwritten.

This is a **warm restart**, not an exact training continuation: the original
files do not include Adam moments. Starting checkpoints can come from different
primary update numbers, but the spent primary budget is 6,000 for every model
and the new budget is 3,000 for every model. All measurements are supplemental
and exploratory, not substitutes for the five-seed primary experiment.

Expansion rule: add seeds **1 and 2**, with the identical three-head protocol,
if seed-0 S4's validation objective improves by more than **5%**, or if the
MLP-versus-S4 ordering of validation raw MSE reverses. Use validation for this
decision, not test metrics. Otherwise stop after the bounded seed-0 check.
Whitened validation objectives have different units from raw objectives, so
head ordering must use the separately recorded raw validation MSE.

## Run after primary MPS training finishes

```bash
.venv/bin/python -m fm_stress.optimization_check \
  --config configs/main.json --results results/main \
  --device mps --seeds 0
```

If the expansion rule fires:

```bash
.venv/bin/python -m fm_stress.optimization_check \
  --config configs/main.json --results results/main \
  --device mps --seeds 1,2
```

Default output is `results/main/optimization_sensitivity/`, containing per-run
metrics, baseline comparisons, learning curves, selected checkpoints, generated
samples, and an aggregate `summary.json`. Each record stores the exact original
checkpoint SHA256, original configuration hash, warm-restart options and selected
additional update. A matching completed run is reused; changed options are not
silently overwritten. `--heads`, `--additional-steps`, `--n-paths`, and `--output`
exist for smoke tests, but the research sensitivity uses the defaults above.
