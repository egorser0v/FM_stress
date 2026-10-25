# Independent presentation visual and scientific audit

Date: 25 October 2026. Current revision: **18 slides, 14 main and four backup**. All 18 current PNGs were individually inspected with the image-viewing tool: `.presentation-build/render/slide-01.png` through `slide-18.png`, render timestamp 22:30, with slides 11 and 12 re-inspected after the 22:31/22:33 wording rebuilds. This is a visual review of exported slides, not a claim of opening the deck in Microsoft PowerPoint. The user-approved role revision replaces the previous contribution placeholder with main slide 14. That new slide was inspected by the parent and novice reviewer; the scientific content of the remaining slides is unchanged.

## Visual result

All slides have readable titles, body text and numeric labels at 1,280×720. No table text, plot legend, slide number or footer is clipped. Long titles on slides 16 and 17 wrap without colliding with content. Scientific plots occupy substantial areas on slides 2, 3, 7, 8, 9, 10 and 11. Axes, legends, forecast boundaries and uncertainty labels are readable; bar charts retain zero baselines.

| Slide | Checked content | Visual/scientific assessment |
|---:|---|---|
| 1 | Title, question, authors, defense date | Clear hierarchy; no overflow. |
| 2 | Forecast task and fixed trajectory example | First history, observed future and four actual forecast draws; boundary and legend readable. Notes state the single-context limitation. |
| 3 | Starting sources and sampling | Comparable axes; notes distinguish unit-variance source illustration from exact main-run covariance. Generation time is distinguished from forecast time. |
| 4 | Main 2×2 experiment | Two velocity heads crossed with white/GP starting noise, five seeds per cell. Whitening is a separate five-run control. History-conditioning and leakage details remain in notes. |
| 5 | Metric meanings | Held-out update-prediction MSE, forecast scores and shape diagnostics are distinguished. |
| 6 | Original white/GP results and all-fail rule | All five cells visible and numerically consistent; no replacement of the original 0/25 result. |
| 7 | R1 versus R2 | S4 R2=1.63 distinguished from R1≈1.03; slope-change magnitude, not event count. |
| 8 | Main score-contrast intervals | Zero references readable; S4 VS upper bound +0.00002 retained. Whitening VS result is scoped to this sample and exploratory analysis. |
| 9 | F64 capacity trajectories | Four matched draws, shared axes and understandable model titles. Footer 19.09/1.60/1.00 agrees with aggregate ensemble R2 across three draws, not the single plotted context. |
| 10 | Main ETTh1 normalization comparison | GP source and 118 windows visible; global/center/history numbers correct. Matched history-ridge 0.0936 and reused-period limitation retained. Rounded equal bars do not establish equivalence; notes explain pipeline scope. |
| 11 | Main vector-valued multichannel comparison | All nine GP-source CRPS bars agree with canonical means. Fixed TRAIN-channel scaling and 2,000 updates visible. Ridge-residual baseline beats displayed heads and is stated on-slide. The slide itself forbids cross-task score ranking and states that channel count is not isolated; its subtitle distinguishes this 64-context cohort from slide 10. |
| 12 | Main response to review | Four rows cover PC3–6, the reference with the same observed history but a known synthetic distribution, fresh synthetic draws, and remaining conditioning/tuning limits. The new reference wording avoids claiming it lacks privileged distribution knowledge; forecast ensembles are already covered in slide 8 and notes. Footer explicitly retains whitening floor/shrinkage and S4 internal-step-range gaps; Euler checks do not resolve them. |
| 13 | Supported conclusions | No universal head/source winner; whitening conclusion scoped to two real multichannel tasks at this budget. Linear-baseline strength and AI assistance remain visible. |
| 15 | Original literal success rule | Correct conjunction; 25/25 roughness passes and 0/25 full successes; privileged-oracle qualification retained. |
| 16 | PC3–6 and matched-oracle discrepancy | Correct values, different-cohort qualification, GP-source label and PC7 transition caveat. |
| 17 | Label-supervision results | GP subset labeled; three paired draws and privileged analytic labels disclosed. Mean improvements do not imply each S4 draw improves. |
| 18 | Fair VS and uncertainty | Correct fair correction; conditional fixed-model intervals distinguished from three paired follow-up replicates; post-hoc status retained. |
| 14 | Team contributions | Four English summaries match the user-approved canonical statements. Readable two-column table; 30 seconds already included in the main-talk budget. AI disclosure retained. |

## Coverage and numerical consistency

Normalization, multivariate prediction and the reviewer response now occur in the main talk rather than only in backup. The notes budget 730 seconds of speech (including the 30-second contribution slide) and 75 seconds of transitions: 805 seconds (13:25), below 15 minutes. This is a planned budget, not a measured rehearsal; backups 15–18 are for questions.

The new paper Section 6 was checked against `output/extension-analysis/summary.json` and `output/sparse-analysis/summary.json`: all 48 neural CRPS/energy table entries agree to displayed precision, as do the eight unique ridge-residual entries. The multichannel slide uses the same GP-source CRPS values. Both-source results and the synthetic three-channel control remain in the paper. Multivariate experiments use 64 contexts ×32 members, 2,000 updates, three optimization seeds and global TRAIN-channel normalization; the normalization intervention uses 118 contexts and 6,000 updates. No pooled cohort or dimension-only causal claim is made.

Scientific wording corrected during earlier passes remains corrected: held-out MSE is not training error; R2 is change magnitude; Euler integration, flow-time conditioning and internal S4 discretization are distinct. Remaining ablations are not presented as completed. No unresolved blocking visual or scientific issue was found in this revision. The user approved the role statements in `docs/contributions.json`; these are not inferred from Git or independently established by the numerical audit.

## Plot provenance and review independence

`forecast_task.png` uses main GP+MLP seed 0, context 0 and members 0–3. `capacity_paths.png` uses first configured draw 607041, context 0 and members 0–3 at 12,000 updates, matching the archived capacity example selection. `source_curves.png` reproduces the prior-mixture illustration with seed 50371 and saved endpoint covariance matrices; it is not a forecast.

This reviewer authored those three plot layouts in `scripts/build_defense_figures.py`, so their inspection is visual QA rather than independent implementation verification. Input/ZIP-entry hashes, selections and output hashes are in `output/final-presentation/figures/provenance.json`, covering only those three PNGs. The separately authored score-interval plot has separate provenance. No context selection search, training run or forecast regeneration was introduced. This pass changed audit documents only, with no deck edits or commits.

## Fresh audience revision

A new GPT-5.6 Sol agent (low reasoning, no inherited chat) reviewed isolated copies of the final slide images and speaking notes. It had no repository or previous-review context. See `output/final-presentation/fresh_audience_review.md` for its questions, factual corrections to the review, and recheck. Wording now separates GP-generated data from GP starting noise, explains the fixed two-sided interpretation of the assignment, retains 0/25 alongside explicitly post-hoc diagnostics, and limits the concluding forecast claim to the main GP-source comparison. Notes explain that PC2 is recorded separately. The review slide states the oracle information advantage directly; capacity repeats refer to three new synthetic datasets.

Scientific chart arrays, embedded figure hashes, the five-cell main result table, author order, approved contribution rows and all four backups are unchanged. Changed visible slides 4–8 and 10–13 were re-rendered and inspected. The shortened notes allocate 730 seconds of speech plus 75 seconds for pauses and handovers (13:25); this remains an estimate, not a measured rehearsal.

## Main factorial-design slide

Slide 4 now shows the two-by-two source/head matrix directly, followed by the additional whitening control and the question about the change in the MLP–S4 gap. The existing fresh-audience reviewer checked this local revision, its arithmetic (20 main runs plus five control runs), and the 84-word English speech. PPTX rendering and PDF page 4 were inspected. Other slides and the 13:25 timing are unchanged.


## Research-narrative revision (25 October, after the matrix correction)

The review now covers the experiment's argument, not only text legibility. Restored the geometry check, original R1 and substantive velocity result; clarified network velocity vs integration, GP-only forecast scores and changing follow-up settings; added synthetic GP3 to multivariate evidence; corrected the whole-MLP capacity intervention. All 18 artifact-tool renders were inspected. Four native charts and four scientific figures remain. See `output/final-presentation/research_narrative_audit.md` for review boundaries, findings and resolutions. Earlier readability approvals did not establish complete coverage of the assignment.


## Superseding first-use check — 26 October

The user identified an unresolved first-use error: R1 appeared before its definition. Earlier approvals must not be read as proof that visible slides were self-contained. A new agent, without conversation context, reviewed only slide images and visible text, then rechecked the revised images. Definitions were added to the slides themselves. Full findings and resolution are in `output/final-presentation/visible_only_review.md`.


## Explicit roughness formulas — 26 October

Slides 6 and 7 now show editable equations for R1 (mean absolute first difference) and R2 (mean absolute second difference), with the correct H−1/H−2 denominators and summation bounds. The symbols x, H and i are defined alongside each formula. Reported ratios explicitly compare means over generated and observed paths. The slides-only reviewer checked both revised images; final PDF export was also inspected. Equations match `fm_stress/trajectory_metrics.py:roughness` and its ratio aggregation.
