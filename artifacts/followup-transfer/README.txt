FOLLOWUP TRANSFER — EXISTING ARTIFACTS ONLY

Code commit: d9ad87495ee16f285e09038a42b1639b6ef3c82a
Branch: additional_expriments
24 models: 12 GP label-noise (6000-step fits), 12 RBF F64_l8 white-source
capacity (12000-step fits). Checkpoints are validation-selected, so their
selected step can be earlier than the completed training budget.

checkpoints.zip: 24 byte-identical original .pt files, corresponding unchanged
run JSON, two saved configurations/manifests/environment records, and the
corresponding repository config files. There are no optimizer states.
forecast_samples.zip: 24 filtered NPZ files containing only generated and
ensemble, plus six shared targets_history.npz files (one per family/data_seed).
No model was executed and no metric or new forecast was calculated to package
these artifacts. Existing run JSON diagnostics are copied, not recalculated.

Extract both archives into the same directory. Checkpoint paths mirror the
repository; exported forecasts use a separate forecasts/ directory so that
partial exports do not overwrite the original full samples archives.

ARRAY ALIGNMENT (zero-based):
  generated[i] <-> path_target[i], path_history[i], i=0..1023.
  ensemble[i,j] <-> ensemble_target[i], path_history[i], i=0..255, j=0..31.
  ensemble_target equals path_target[:256].
  Ensemble history is path_history[:256], with no duplicated history array.
All four models in a family/data_seed use the same shared targets/history.
There is no correspondence assumed across data_seed values.
The generated draw is not necessarily ensemble member zero.

All arrays are in the original history-normalized coordinates. Generated and
ensemble are original float32; targets/history are original float64. No
rounding, dtype conversion, subsampling or regeneration was performed.
Raw per-window location/scale are not present in the selected source fields;
they are not reconstructed or supplied. See manifest normalization for the
exact normalization formula and the limitation on conversion to raw units.
CoordinateModel q/scale buffers are embedded in every state_dict (identity/unit
for these models); no external PCA transform is required for model loading.

LOADING WITH YOUR EXISTING BRANCH CODE (no inference required):
  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
  Load the config from results/<family>/config.json in checkpoints.zip.
  For label-noise, use fm_stress.followup_label_noise.build with
  GPConfig(**config["data"]), head, seed and torch.device("cpu").
  For capacity, use fm_stress.followup_capacity.build_model with a
  ScanGeometry constructed from the run's geometry fields (omit derived ratio),
  head, seed and torch.device("cpu").
  model.load_state_dict(checkpoint["state_dict"]); model.eval().

manifest.json records original source paths/hashes, transferred entry hashes,
array shapes/dtypes, row alignment, generation parameters/seeds, selected steps
and completed budgets. Original .pt bytes and exported generated/ensemble NPY
member bytes are preserved exactly. Capacity shared NPY members are also copied
unchanged; label target/history members contain exact slices of test.npz.
No full test file is included. Missing files: none.

SHA256SUMS covers both archives, manifest and this README. The manifest identifies
the exact source-code commit; use git log -- artifacts/followup-transfer for the
later packaging commit, which cannot contain its own hash.
