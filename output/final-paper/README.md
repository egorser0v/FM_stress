# Final scientific paper

`main.pdf` is the ten-page English paper, including references and a reproduction appendix. `main.tex` is its editable source. The title is **Smooth Sources in Conditional Flow Matching: A Controlled Comparison of MLP and S4 Heads**.

`evidence.json` is the full-precision canonical evidence for paper and slides. It includes the original 25 runs, forecast evaluation, capacity, normalization, label supervision, solver sensitivity, PCA audit, selected mixture/objective controls, the archived multivariate extension, and the final 39-ensemble trajectory analysis. `source_paths` maps consumed repository-relative paths to SHA256. Repeated budget checkpoints are not independent fits. Compute entries describe logged run/training durations and explicitly do not claim total elapsed or dedicated GPU hours.

`build_manifest.json` binds the numerical sources, evidence, editable manuscript and vector figures. The parent delivery metadata in `FINAL_PACKAGE.json` and `docs/final_package.md` identifies the final frozen revision and package checks.

From the repository root, with requirements installed:

```sh
python scripts/build_final_paper.py
```

The builder regenerates evidence, all nine tables, the approved author contributions, numerical macros and two vector figures, then runs pdfLaTeX twice. It requires Python with NumPy/Matplotlib and a `pdflatex` executable. `--no-compile` regenerates evidence/assets only. An existing LaTeX editor can also compile `main.tex` directly after the generated files are present. The local verification used the bundled Python runtime with plotting dependencies under `/tmp/fm-stress-plot-deps`; this temporary path is not required when the project's requirements are installed.

Scientific audit corrections are incorporated: original flags stay 0/25; the privileged and information-matched references are distinct; primary path and ensemble cohorts remain distinct; variance-floor directions are not equated with signal-free directions; pipeline-matched ridge remains visible; intervals spanning zero do not imply equivalence; R2/variogram diagnostics do not create a new winner. The author-contribution paragraph reproduces the user-approved statements in `docs/contributions.json`; `contributions.tex` is generated from that source. Oral preparation remains the team's responsibility.

The multivariate section uses all four archived datasets (3-channel GP, ETTh1 OT, 7-channel ETTh1, 8-channel Exchange), both sources and MLP/S4/whitened MLP. Both marginal CRPS and joint energy / sqrt(F*C) retain the direct ridge-residual baseline. The protocol uses global TRAIN-channel scaling, 2,000 updates, three optimization seeds and 64 contexts × 32 members; it is separate from the original history-normalized experiment. `evidence.multivariate.cells` contains 24 aggregate cells (72 archived checkpoints), with protocols, dataset baselines, archive counts and independent audit summaries alongside it. The complete underlying suites contain 96 extension fits plus 288 additional sparse/structural-control fits; this paper revision trains no models.
