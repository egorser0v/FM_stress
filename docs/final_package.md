# Final defense package — 26 October 2026

## Deliverables

- [Conference-style paper, 10 pages](../output/final-paper/main.pdf), with [editable LaTeX](../output/final-paper/main.tex).
- [Editable presentation](../output/final-presentation/defense.pptx) and [PDF](../output/final-presentation/defense.pdf).
- [Russian speaking notes](../output/final-presentation/speaker_notes_ru.md): 14 main slides: 730 seconds of speech (including 30 seconds for contributions) + 75 seconds for transitions = 13:25 planned total, below the 15-minute cap. Four backup slides are for questions only. This is a planned allocation, not a measured rehearsal. Normalization, multichannel results and the explicit review response are all in the timed talk.
- [Canonical evidence](../output/final-paper/evidence.json), [trajectory table](../output/trajectory-metrics/metrics_table.md), and [review response](response_to_review.md).
- [Scientific audit](final_scientific_audit.md), [independent metric audit](trajectory_metrics_independent_audit.md), and [novice presentation review](../output/final-presentation/novice_review.md).

**Contributions approved:** the user-approved English statements in `docs/contributions.json` appear in the paper and on main slide 14. They describe the agreed division of research work, not ownership inferred from Git. Rehearse the talk and submit the materials to the mentor by 26 October at 12:30 Moscow time; the package does not claim that submission has occurred.

## Frozen analysis revision

The research/analysis revision is `514ab01e967be1eef49857716f406ebc86c08799` on `main`. It contains the colleague's follow-ups and transferred archives plus three post-hoc commits. `FINAL_PACKAGE.json` pins this revision and the delivered files by SHA256. The later package commit adds manuscript, slide sources and this index; its identity is available from Git history, avoiding a self-referential commit hash inside its own contents.

There was **no new model training or forecast generation** in the post-hoc metric contribution. It reads 15 main GP ensembles and 24 transferred ensembles in place. Full saved single paths are a separate cohort. The fixed original success rule remains unchanged (0/25 passes). No test metric selected a new checkpoint. Follow-ups and new score comparisons are exploratory.

## Minimal post-hoc reproduction

Python 3.12 was used. Only NumPy is required to recompute the archived scores. From the repository root:

```sh
python -m venv .venv
.venv/bin/python -m pip install numpy==2.5.4
.venv/bin/python scripts/evaluate_trajectory_metrics.py
.venv/bin/python scripts/analyze_roughness_posthoc.py
.venv/bin/python scripts/audit_final_pca.py
```

Inputs are versioned, including the compact ZIP transfer. No checkpoint extraction or download is needed. Re-running may rewrite the new output files; archived experiments are read-only. Numerical outputs are reproducible; ZIP/NPZ and PDF container metadata can vary without changing the stored values. Input provenance includes source-file and transferred-entry hashes.

For figures and paper:

```sh
.venv/bin/python -m pip install matplotlib==3.11.2
.venv/bin/python scripts/plot_roughness_posthoc.py
.venv/bin/python scripts/plot_trajectory_metrics.py
.venv/bin/python scripts/build_final_paper.py
.venv/bin/python scripts/build_defense_figures.py
.venv/bin/python scripts/build_defense_score_figure.py
.venv/bin/python scripts/build_defense_content.py
```

The paper requires pdfLaTeX (TeX Live with the packages listed in `main.tex`). `--no-compile` generates the tables/evidence/figures without TeX. Slides are editable without their builder. Rebuilding the PPTX uses Node.js and the Codex bundled `@oai/artifact-tool` 2.8.99; set `ARTIFACT_RUNTIME_ROOT` and `PRESENTATIONS_SKILL_DIR` for another installation, then run `node scripts/build_final_presentation.mjs`. This proprietary authoring dependency is not needed to reproduce any scientific score. The checked-in content JSON and speaker notes are plain text. PDF export used LibreOffice Impress.

For full model reproduction, use [the original README](../README.md), `pyproject.toml`, the archived `requirements-lock.txt`, the exact `configs/` files and the study protocols. The original lock describes the experiment environment; the present local verification used Python 3.12.14, NumPy 2.5.4, PyTorch 2.14.1, SciPy 1.18.1, pytest 9.1.1, and CPU execution. Do not silently treat this verification environment as the original training environment.

## Data, split, seeds and compute

- Main synthetic geometry and split: [protocol](protocol.md), `configs/main.json`, `results/main/manifest.json`. Independent windows, history-only normalization, train-only PCA. Data seed 27041; optimization seeds 0–4.
- Main fresh-history ensemble evaluation: [protocol](ensemble_evaluation_protocol.md), `results/ensemble-evaluation/config.json`, 256 histories × 64 members, Euler128. Original 1024-path diagnostics used Euler64.
- Selected label supervision follow-up: data seeds 917041/1017041/1117041 paired with training seeds 81/82/83; GP source, 12 models, budget6000.
- Selected capacity follow-up: data seeds 607041/707041/807041 paired with training seeds 61/62/63; RBF, horizon64, white source, 12 models, budget12000.
- Both selected follow-ups: 256 histories × 32 members, Euler64, validation-selected checkpoints. The transfer manifest gives exact selected steps, original configs, hashes and all shapes. Budgets are not selected checkpoint steps.
- ETTh1 source/split/preprocessing: [protocol](followup_normalization_protocol.md) and `configs/followup_normalization.json`. Raw ETTh1 data are not in this clone; recorded result tables are available. This is one time period, not an independent multi-dataset validation.
- Compute: `evidence.json:compute` records unique fit counts, timing scopes and summed logged durations. The colleague trained 144 follow-up fits on MPS. The archived multichannel/sparse suites contain 96 + 288 fits; the final paper extracts 24 MLP/S4/whitened aggregate cells (72 archived checkpoints), not a fresh training contribution. Concurrent durations are not dedicated GPU-hours or end-to-end wall time. The new score analysis is CPU-only and requires no training.

- Multichannel extension: `configs/extension.json` and `configs/sparse_extension.json`, three optimization seeds 31–33, 2,000 updates, 64 held-out contexts × 32 members, Euler64. Fixed TRAIN-channel scaling, not per-history normalization. Data: 3-channel GP, ETTh1 OT, 7-channel ETTh1 and 8-channel Exchange. Canonical source summaries are `output/extension-analysis/summary.json` and `output/sparse-analysis/summary.json`; `evidence.multivariate` retains their values, baseline metrics and provenance. Joint and single-channel tasks do not isolate dimension causally.

## Checks and interpretation limits

The merged suite passed **218 tests**, with **5 skips** for unavailable MPS tests and the unfetched ETTh1 CSV. Twelve new trajectory tests include exhaustive unbiasedness and explicit pairwise score calculations. The independent audit checked 1,877 numeric comparisons and 90 input hashes, with maximum discrepancy 3.55e-15. The evaluator reproduces all 156 archived consistency fields.

R2 measures average absolute second differences, not coefficient of determination. Variogram scores use exponent0.5 and normalized inverse-lag weights over unordered time pairs. The finite-ensemble correction assumes conditionally independent members. Main intervals condition on five fixed trained models and shared test contexts. Three follow-up draw/seed replicates do not justify a precise population interval. No new pass threshold or universal model winner is claimed.

AI assisted coding, numerical checks, scientific review and drafting. Separate agents are not independent human validation. Named authors retain responsibility for the final claims and oral defense. Commit authorship reflects the user's requested Git identity; it is not evidence that every line was manually written.
