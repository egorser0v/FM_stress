# Smooth source, unstructured head

Reproducible, small-scale conditional flow-matching experiments for the supplied
course project. The implementation tests **white Gaussian vs matched-kernel GP
sources**, **AdaLN MLP vs three-block S4-NPLR velocity heads**, and a mandatory
**PCA-whitened MLP control**. This is not pretrained Sundial or a full TSFlow
benchmark reproduction.

## Start here

- Poster: [PDF](output/poster/flow-matching-poster-LaTeX-A1.pdf) · [editable LaTeX](output/poster/flow-matching-poster.tex). This is the project proposal; completed results are in the reports below.
- Reports: [main experiment](output/pdf/research-report.pdf) · [S4/MLP follow-up](output/pdf/s4-mlp-followup.pdf) · [related work](output/pdf/related-work.pdf).
- `docs/README_RU.md`: a short Russian guide for the team.
- `output/pdf/research-report.pdf`: completed experimental report.
- `output/pdf/s4-mlp-followup.pdf`: additional head-by-loss and conditional-forecast experiments.
- `output/pdf/related-work.pdf`: the required one-page critical synthesis.
- `docs/related_work.md`: critical literature synthesis and official sources.
- `docs/protocol.md`: data generation, normalization, metrics, success rule and
  interpretation limits, declared before the scored experiments.
- `docs/model_provenance.md`: exact upstream revisions, licenses, and adaptations.
- `docs/requirements_matrix.md`: each assignment task and its deliverables.
- `configs/main.json`: complete scored experiment settings.
- `results/main/`: per-seed measurements, best checkpoints, training curves and
  example/generated paths. `results/pilot/` is development data, not final evidence.

## Primary findings

Across five matched seeds at the fixed 6,000-update budget, the predicted GP-source
MLP disadvantage against the tested S4 head did not appear. GP+MLP residual MSE was
0.04927 ± 0.00077 versus 0.05827 ± 0.00117 for GP+S4. Regularized whitening reduced
the MLP's residual MSE to 0.04010 ± 0.00028 (18.6% lower) and its PC1 MSE by 8.8%.
All 25 primary runs met the roughness band; none met the literal two-sided raw
error-equality condition. The privileged analytic reference also fails that
equality condition. These observations support further work on coordinates,
not a universal architecture ranking or a failure claim about pretrained models.

The original final report separates the primary experiment from the shorter-lengthscale
follow-up and the lower-learning-rate warm restarts. `docs/conclusions.md` gives
the qualified recommendation across those checks.

## Additional S4 / MLP checks

The original measurements above are retained. The follow-up separates two questions:

- `configs/controlled_followup.json`: fresh GP data, identical full-history identity
  conditioning, MLP/S4 × raw/balanced velocity loss, eight validation-only learning-rate
  pilots and twelve final fits. Balanced loss weights training-PCA errors but leaves
  both networks in temporal coordinates.
- `configs/ensemble_evaluation.json`: the fifteen original GP checkpoints evaluated
  on 256 fresh histories with 64 samples each; fair CRPS, path energy score, marginal
  coverage/width and a paired Euler-128/256 sensitivity check.

Protocols and independent review are in `docs/controlled_followup_protocol.md`,
`docs/ensemble_evaluation_protocol.md` and `docs/followup_independent_audit.md`.
The final interpretation is in `docs/followup_conclusions.md`. Small-direction
diagnostics are in `output/field-diagnostics/`. The original pass flags are not
replaced by these supplementary comparisons.

The new controlled experiment shows an objective-dependent result: normalized
residual error favors MLP with raw loss (0.222 vs 31.122), but S4 with balanced
loss (0.158 vs 0.175). Balancing removes S4's excessive generated tail energy but
worsens its mean raw errors; MLP still has lower raw residual error under both
losses. For the original checkpoints on fresh histories, neither S4 nor whitening
has a resolved CRPS/energy advantage over MLP: all four paired-history intervals
include zero. All 34 tests passed, including CPU/MPS checks, and an independent CPU audit recomputed
every new scored result from saved arrays and verified provenance.

Run the heavy MPS jobs **sequentially**, then verify and analyze:

```sh
.venv/bin/python -m fm_stress.controlled_followup --device mps
.venv/bin/python -m fm_stress.analyze_controlled
.venv/bin/python -m fm_stress.ensemble_evaluation --device mps
.venv/bin/python -m fm_stress.analyze_ensemble
.venv/bin/python scripts/check_followup.py
.venv/bin/python scripts/build_followup_report.py
.venv/bin/python scripts/package_project.py --extended
```

The extended package is `output/fm-stress-extended.zip`; it includes both the original
project and the new checkpoints, sample arrays, analyses and verification records.
Saved source hashes reject accidental mixing of numerical code revisions.

## Environment

Python 3.12 on macOS/Apple Silicon is the tested environment. Create an isolated
environment and install the exact dependencies:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

Or use `uv venv --python 3.12 .venv` and
`uv pip install --python .venv/bin/python -r requirements-lock.txt`.

The original machine has an Apple M3 Pro and 18 GiB unified memory. MPS must be
visible to the process; some application sandboxes hide the GPU. The runner
**fails rather than silently substituting CPU** when `--device mps` is unavailable.
Pass `--device cpu` explicitly to reproduce on CPU. GPU floating-point execution
is not guaranteed bitwise reproducible across PyTorch/macOS versions.

## Reproduce

Run from this directory:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m fm_stress.experiment geometry --config configs/main.json --output results/main
.venv/bin/python -m fm_stress.experiment train --config configs/main.json --output results/main --device mps
.venv/bin/python -m fm_stress.experiment oracle --config configs/main.json --output results/main
```

For all experiment, diagnostic, analysis and report steps, run
`bash scripts/run_all.sh`. To rebuild tables and figures from saved runs only:

```sh
MPLCONFIGDIR=/tmp/fmstress-mpl .venv/bin/python -m fm_stress.analyze --results results/main --output output/analysis --require-complete
MPLCONFIGDIR=/tmp/fmstress-mpl .venv/bin/python scripts/build_report.py
MPLCONFIGDIR=/tmp/fmstress-mpl .venv/bin/python scripts/check_delivery.py
.venv/bin/python scripts/package_project.py
```

The report builder checks completion and configuration consistency. Its authored
conclusion is in `docs/conclusions.md`; after changing an experiment, review this
interpretation rather than assuming that regenerating tables rewrites the claims.

Completed runs with the same config are skipped. To run changed code or settings,
use a **new output directory**; do not mix checkpoints from different experiments.
Individual cells are selectable, e.g. `--cell gp:s4 --seed 0`. All checkpoint
selection uses validation loss. Final test windows, interpolation times and source
draws are shared across cells to make comparisons paired. All five main training
seeds are reported; no seed is selected by test performance.

The main series uses 32,768 training, 4,096 validation and 8,192 test windows;
1,024 test paths per cell; 6,000 optimizer steps, batch 256; and 64 Euler steps.
The models have 65,464 (MLP) and 65,281 (S4) trainable parameters, under 0.3%
difference. Encoder architecture and initial weights match; encoder weights then
train separately with each head. Models train on true velocity labels `y-epsilon`,
even though the released Sundial software uses a different target-prediction loss.

## What the success rule does and does not mean

The assignment's literal two-sided 20% criterion is retained in `metrics.success`:
residual-PC MSE / PC1 MSE and generated / target roughness must both be in
[0.8, 1.2]. Raw residual MSE averages PCs 3..F; PC2 is reported separately.
Because the geometry is highly anisotropic, very low residual error can fail this
equality criterion. Do not present `works=False` as proof that a model cannot
forecast. Scale-normalized diagnostics and a privileged conditional-GP oracle are
reported separately, without replacing the primary criterion after seeing results.

The oracle sees full raw history, while the networks see normalized history, so
its error floor is an informational lower bound, not necessarily attainable by a
network. Its reference roughness uses exact GP conditional draws; it is not the
roughness of an Euler-integrated learned model.

## Repository layout

```text
fm_stress/          data, metrics, actual models, training and analysis
configs/            fixed pilot/main and robustness configurations
tests/              analytical, leakage, parity and integration tests
vendor/             licensed upstream reference code + immutable SHA manifest
docs/               protocol, literature, provenance and requirements
results/            raw evidence, logs, checkpoints and samples
output/             final tables, figures, reports and the editable poster
```

The existing `output/poster/flow-matching-poster.tex` remains the editable proposal
poster. Its old PowerPoint export is historical and does not match the current
LaTeX. Experimental measurements belong in the results report, not in an invented
backfilled proposal.

## Attribution

The AdaLN architecture is adapted from the official
[Sundial implementation](https://huggingface.co/thuml/sundial-base-128m).
The S4 state-space construction follows the official
[state-spaces/s4](https://github.com/state-spaces/s4) code, with real arithmetic
and a finite-horizon realization for MPS. The three-block temporal backbone is
inspired by [TSFlow](https://github.com/marcelkollovieh/TSFlow).
See `vendor/manifest.json` and the preserved licenses before redistribution.
