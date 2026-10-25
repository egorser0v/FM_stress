# Research-narrative audit, 25 October 2026

The earlier readability checks missed a structural problem: the deck listed components and controls without preserving the original experiment as the organizing question. This pass checked each slide against the assignment, protocol and saved evidence, and then checked the complete spoken sequence. It supersedes earlier blanket claims that the presentation had no remaining content problems.

## Review boundaries

Three separate agents performed bounded reviews: original-assignment and research-argument review; evidence/code review; and audience/story review without repository or conversation context. The latter received only the deck, notes and speaking scripts. These are AI checks, not independent human scientific validation or a measured rehearsal.

The primary agent reconciled the findings, edited the deck, viewed all 18 rendered slides, and retained the original plots. The evidence reviewer rechecked the revised content against frozen outputs and source code. The argument reviewer reread main slides 1–14 and the notes as one sequence. Local English scripts were rewritten to match. No experiment was rerun and no result was selected or removed to improve the apparent outcome.

## Findings and corrections

| Problem | Correction | Evidence or check |
| --- | --- | --- |
| A list of models, whitening and leakage controls obscured the experimental design. | Slide 4 now shows the original two heads × two sources, with whitening as a separate five-run control. | `docs/protocol.md`; main registry |
| The network's output was confused with the generation procedure. | Slide 3 separates velocity target, network inputs/output and numerical integration. One random start produces one trajectory. | `fm_stress/experiment.py`; training and sampling code |
| The original geometry gate was absent. | Slide 5 displays held-out top-two-PC variance for target curves, white-source velocities and GP-source velocities. Main history normalization and train-only PCA are explicit. | `evidence.main.geometry`; code computes variance, not uncentered energy |
| PCA errors could be mistaken for forecast errors; only the failed rule was emphasized. | Slide 6 identifies velocity errors, shows all original cells plus the coordinate control, adds original R1, and states the observed GP residual-error ordering. The declared 0/25 outcome remains. | `evidence.main.groups`; each run's rule outcome |
| Original R1 and added R2 were described as if both were new diagnostics. | The original table uses original R1; slide 7 retains R1 and adds R2 in a separate ensemble rescore. | Original 1,024-path evaluation vs 256 histories × 64 draws |
| A broad forecast interaction was implied by GP-only forecast scores. | The question concerns the predicted velocity-head failure. Slides 7–8 explicitly identify GP-source forecasts. | Saved main ensemble uses GP checkpoints |
| Capacity intervention was described as widening only the output. | Slide 9 specifies whole-MLP width, white source, history/horizon and budget. Notes separate the same-depth and near-parameter-matched controls. | L128/F64; 12,000 updates; width 56→72; 73,960→114,328 parameters |
| Follow-up settings appeared to be continuations of one unchanged experiment. | Transitions explain the next question; captions identify changed source, data, normalization and budget. | Capacity, normalization and multivariate protocols |
| Multivariate evidence omitted the synthetic positive whitening result. | Slide 11 now includes synthetic GP3, ETTh1 OT1, ETTh1 7 and Exchange8. The conclusion is task-dependent, not a claim that dimension alone causes failure. | All 12 plotted CRPS means and 4 ridge values checked; joint energy confirms whitening directions |
| Review responses sometimes named a method without answering the question. | Slide 12 reports actual findings, including GP PC3–6 ordering and forecast-score results; remaining limitations are brief. | PCA audit; paired main score contrasts; fresh-draw capacity checks |
| The conclusion did not answer the engineering choice. | Slide 13 recommends retaining MLP as the baseline rather than switching solely because the source is smooth. | Main velocity result plus scoped forecast and transfer checks |

## Final consistency checks

- Main numerical evidence matches the paper and frozen JSON; no cross-cohort scores are pooled.
- The original R1 means were not replaced by the post-hoc ensemble means.
- PCA bases are source-specific and fitted on training velocities; PC3–16 is an average.
- All multivariate heads and the ridge sampler use the same saved evaluation contexts within a task.
- New figures are native editable charts; existing four scientific figure images retain their hashes.
- Main talk remains 14 slides, with 4 backups. Budget: 730 seconds of speech + 75 seconds of pauses and handovers = 805 seconds (13:25). Timing is planned, not measured.
- The updated English rehearsal files remain local and uncommitted, as requested.

Numerical verification is automated in `scripts/verify_final_package.py`. PPTX structure/native-chart validation and PDF export checks are recorded in `qa.json`; final hashes are in `FINAL_PACKAGE.json`.


## Superseding first-use check — 26 October

The user identified an unresolved first-use error: R1 appeared before its definition. Earlier approvals must not be read as proof that visible slides were self-contained. A new agent, without conversation context, reviewed only slide images and visible text, then rechecked the revised images. Definitions were added to the slides themselves. Full findings and resolution are in `output/final-presentation/visible_only_review.md`.
