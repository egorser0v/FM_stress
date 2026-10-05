# Presentation and defense notes

This is a speaking outline, not a script to read verbatim. The task requires every participant to speak and answer questions. Each speaker should understand the common question, the three metrics, the whitening control, and the limits of the conclusion. Use the completed report and saved experiment tables for numerical results; do not quote a partial training run as the final result.

## Four speakers, one argument

### Egor Serov — the question and controlled experiment (about 60 seconds)

“Our question is whether a smooth starting distribution interacts badly with an unstructured velocity network. Sundial’s TimeFlow head and TSFlow motivate the comparison, but their published systems differ in several ways. Comparing their benchmark scores would not isolate the question.

We therefore train small models from scratch in one synthetic setting. We cross two sources, white noise and a smooth GP draw, with two heads, an adaptive-layer-normalized MLP and a temporal S4 network. Everything predicts the same velocity target, the difference between the future patch and an independently sampled source. A tiny encoder summarizes the past. It has the same architecture and initial weights across cells, although each model subsequently trains its own copy.

The MLP and S4 models have 65,464 and 65,281 parameters respectively, a difference below 0.3%. The recorded main configuration uses five seeds, 6,000 updates per model and the same batch size. Our claim is a hypothesis about this controlled geometry, not a claim that pretrained Sundial fails.”

**Point to:** the source-by-head table, then the saved configuration and parameter counts. **Handoff:** “Kirill will explain how we verify that the data actually instantiate the proposed stress test.”

### Kirill Frolov — data and the geometry gate (about 60 seconds)

“We draw a 32-point history and a 16-point future jointly from one Gaussian process. Each example is an independent window. We normalize the history using its own mean and population standard deviation and apply those same statistics to the future. Future values never determine the normalization.

The two sources are independent of the target. The GP source uses the same raw kernel as the data generator. After context normalization, however, the target’s marginal covariance is not exactly that kernel; we disclose this distinction.

Before training, we fit PCA on training examples and check the spectra on validation examples. The matched-GP velocity must have more than 90% of its variance in the first two directions. We also check that the paths are not constants and that white-source velocities retain a contrasting spectrum. A separate test draw supplies the reported held-out spectrum. The main data use different seeds from the pilot.”

**Point to:** the geometry table and example paths. **Handoff:** “Daniil will explain what the temporal model is and how we judge a generated path.”

### Daniil Koblov — S4 and evaluation (about 75 seconds)

“The temporal model uses three bidirectional S4 residual blocks. It is a small TSFlow-style adaptation: we reduce the internal state dimension to 16 and use the common history encoder. It remains an S4 state-space kernel with a learned low-rank correction, rather than an arbitrary convolution or an S4D substitute.

For this short horizon, we compute the finite state-space kernel using real matrices on Apple MPS. We checked its outputs and gradients against an independent complex-valued implementation. This validates the numerical adaptation, not equivalence to the entire published TSFlow system.

We report three required quantities: velocity error along PC1, average velocity error along PCs 3 to 16, and the average adjacent jump in generated paths after 64 Euler steps. PC2 is reported separately. We retain the assignment’s literal 20% criterion, but it has a limitation: requiring residual and PC1 errors to be almost equal can reject a model whose residual error is much smaller. We therefore show raw metrics, variance-normalized diagnostics and an analytic GP reference, without changing the primary rule after seeing results.”

**Point to:** the three-metric table, then the reference row and roughness comparison. **Handoff:** “Vasilii will explain the coordinate control and what the evidence permits us to conclude.”

### Vasilii Lyamin — whitening and the recommendation (about 60 seconds)

“Our required control runs the same MLP with a linear whitening transform fitted to training GP velocities. We transform inputs and velocity targets consistently and map predictions back before evaluation. We run this control whether or not an initial MLP disadvantage appears.

Whitening changes coordinate scaling and the training objective’s emphasis. It does not discard dimensions in this implementation. An improvement would therefore support a useful coordinate intervention, but would not by itself prove that unused channels caused the original behavior.

In the completed five-seed primary experiment, whitening reduced the GP MLP’s residual error by 18.6% and PC1 error by 8.8%, while retaining target-like roughness. The tested S4 did not show the predicted advantage. However, all 25 runs failed the literal error-equality rule. The separate analytic reference also failed that condition. We therefore withhold a method verdict, keep MLP as the baseline, and recommend investigating whitening with a prospectively clarified criterion. The additional optimization and geometry checks are reported separately.”

**Point to:** the whitening row and the report’s recommendation. Explain that lower velocity errors are observed evidence, while the all-fail rule prevents a formal method verdict.

## Difficult questions and concise answers

**Why do independently paired arrows not simply cancel?**
The learned field is the conditional expectation of the sampled velocity given the current interpolant, flow time and available context. Arrows that cross a location can differ, but their conditional mean transports the intermediate distributions. Averaging random training labels is not an identifiability failure. It does create irreducible regression noise. Also, a lookback does not uniquely determine a random future; we do not assume that it does.

**Does low-rank velocity covariance prove that the MLP is poorly conditioned?**
No. It verifies the proposed geometry. Establishing an optimization mechanism would additionally require evidence such as gradient or Jacobian diagnostics and controlled interventions. Our architecture and whitening comparisons can support a local design decision, not prove that mechanism alone.

**Why not use forecast MSE alone?**
A mean forecast can look accurate while generated paths have the wrong variation. Velocity errors test the fitted field in dominant and residual directions, and adjacent-jump roughness directly tests path variation. Even these three metrics do not establish full conditional calibration; that remains a limitation.

**Why omit PC2 from the residual group?**
The assignment defines residual directions outside the first two PCs. We follow that definition and report PC2 separately, so its error is not hidden. Residual MSE is averaged per component rather than summed over 14 components.

**What is wrong with the 20% rule?**
We interpret “within 20%” literally as a two-sided error-ratio band from 0.8 to 1.2. Different PC variances make raw errors naturally unequal; much smaller residual error can fail this rule. The criterion is therefore not a universal measure of good forecasting. Normalized diagnostics are labeled secondary and do not replace the recorded primary criterion.

**What does the analytic reference actually show?**
In the verified `results/main/oracle.json`, analytic-velocity residual/PC1 error ratios are about **0.01543 for white noise** and **0.003987 for the GP source**, both far outside the literal band. Exact conditional GP samples have a roughness ratio about **0.97983**, inside the desired range. This demonstrates a criterion limitation. The normalized GP error ratio is about **1.17215**, illustrating sensitivity to metric scaling; it is not permission to select a preferred rule retrospectively.

**Is that reference directly comparable to the learned networks?**
Only with qualifications. Its velocity uses the full raw history, including information lost by normalized-history inputs, so it is a privileged lower-bound diagnostic. Its roughness comes from exact conditional GP sampling, not Euler integration of the analytic field. It is a composite reference, not a trained model or a demonstration that Euler-64 error is negligible.

**Does “matched GP” mean identical source and target distributions?**
No. It means the source shares the raw generator’s kernel, variance and nugget. Context normalization changes the target’s unconditional distribution. We keep the source independent in normalized observation coordinates and explicitly report this distinction.

**How do you prevent leakage?**
Independent synthetic train, validation and test draws; context-only normalization; train-only PCA and whitening; validation checkpoint selection; a separate main data seed from the pilot. Test spectra describe the selected geometry. They are not used to fit transforms or choose the main configuration.

**Is the comparison fair if the encoders train separately?**
Their architecture and starting weights are identical, and training draws and budgets are paired. Their final weights may differ because of interaction with the head. We disclose that limitation. This tests complete small encoder–head combinations, not an isolated final layer with a frozen shared representation.

**Why not import every published component unchanged?**
The assignment requests a small controlled velocity-regression experiment. We reuse the official AdaLN design and S4 mathematical construction, record revisions and adaptations, and test the MPS implementation. In particular, the released Sundial loss predicts targets with horizon weights, whereas this assignment requests velocity regression. We follow the assignment and do not call it an exact Sundial reproduction.

**Why no real-data benchmark?**
The real-series extension is optional and requires the same geometry gate. The required synthetic experiment and its control are the main deliverables. The final report omits real data, so the conclusion remains about synthetic GP geometry; no real-world forecasting claim follows.

**How was AI-generated work checked?**
The team must understand and explain the implementation. Tests check known-value metrics, GP conditioning, source independence, whitening derivatives, actual runner pairing, and S4 numerical outputs and gradients. Saved configurations, raw results and checkpoints allow inspection. Passing tests supports correctness of those properties; it does not prove the hypothesis.

## Final rehearsal checks

- Quote only the completed run set and state how many seeds finished.
- Distinguish the proposal’s predicted outcome from measured findings.
- Keep the primary experiment and any shorter-lengthscale repair experiment separate.
- Make the recommendation match the three metrics and their uncertainty; do not infer equivalence from a nonsignificant difference.
- Each participant should answer at least one question outside their implementation role.

Evidence locations: `docs/related_work.md`, `docs/protocol.md`, `docs/model_provenance.md`, `configs/main.json`, `results/main/geometry.json`, `results/main/oracle.json`, and the completed main/robustness result tables. The numerical statements above refer to the complete primary table; use the final report for the separately labeled follow-up checks.
