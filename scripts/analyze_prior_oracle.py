"""Privileged analytic GP references for the frozen source-mixture study.

The reference observes full raw history and its normalization statistics. It is
not an attainable Bayes oracle for models receiving normalized history alone.
No model is selected or changed by these test-set diagnostic calculations.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.data import (GPConfig, sample_dataset, conditional_target,
                            conditional_oracle_velocity, conditional_oracle_noise)
from fm_stress.experiment import load_config, save_json
from fm_stress.prior_mixture import (array_hash, file_hash, fixed_pairs, identities,
                                    source_covariance, source_name)


def pc_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"overall_mean": float(values.mean()), "pc1": float(values[0]),
            "pc2": float(values[1]), "residual_mean": float(values[2:].mean()),
            "per_pc": values.tolist()}


def error_decomposition(prediction, oracle, label, axes):
    """Finite-sample identity; the empirical cross term is retained, not dropped."""
    a = (np.asarray(prediction, float) - oracle) @ axes.T
    b = (oracle - label) @ axes.T
    total = np.square((np.asarray(prediction, float) - label) @ axes.T).mean(0)
    discrepancy = np.square(a).mean(0)
    oracle_error = np.square(b).mean(0)
    cross = (2 * a * b).mean(0)
    error = float(np.max(np.abs(total - discrepancy - oracle_error - cross)))
    if error > 2e-10 * max(1., float(np.max(np.abs(total)))):
        raise AssertionError("Squared-error decomposition failed")
    return {"model_label_mse": pc_summary(total),
            "model_oracle_mse": pc_summary(discrepancy),
            "oracle_label_mse": pc_summary(oracle_error),
            "signed_cross_term": pc_summary(cross),
            "identity_max_absolute_error": error}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/prior-mixture")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/diagnostics/prior_mixture_oracle.json")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.input / "config.json")
    manifest = load_config(args.input / "manifest.json")
    for name, expected in manifest["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Frozen numerical source changed: " + name)
    expected_ids = identities(cfg)
    pending = []
    records = []
    for d, alpha, head, seed in expected_ids:
        run = f"{source_name(alpha)}_{head}_seed{seed}"
        path = args.input / f"dataset_{d}/runs/{run}.json"
        if not path.exists():
            pending.append({"data_seed": d, "run": run})
        else:
            record = load_config(path)
            if record["status"] != "complete" or record["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Incomplete or incompatible record: " + str(path))
            records.append((record, path))
    if pending and not args.allow_partial:
        raise ValueError(f"Require all {len(expected_ids)} runs; missing {len(pending)}")
    cells = []
    arrays = {}
    for d in cfg["data_seeds"]:
        geometry_path = args.input / f"dataset_{d}/geometry.json"
        geometry = load_config(geometry_path)
        gp = GPConfig(**cfg["data"])
        data = sample_dataset(gp, cfg["n_test"], d + 2)
        mean, covariance = conditional_target(gp, data)
        target_axes = np.asarray(geometry["target_pca"]["components"])
        for alpha in cfg["alphas"]:
            source = source_name(alpha)
            x, label, time, _ = fixed_pairs(cfg, data, alpha, d, "test")
            source_cov = source_covariance(cfg, alpha)
            oracle = conditional_oracle_velocity(x, time, mean, covariance, source_cov)
            noise = conditional_oracle_noise(time, covariance, source_cov)
            if not np.isfinite(oracle).all() or not np.isfinite(noise).all():
                raise ValueError("Nonfinite analytic reference")
            bases = {"source_velocity_pca": np.asarray(geometry["sources"][source]["pca"]["components"]),
                     "fixed_target_pca": target_axes}
            basis_results = {}
            for name, axes in bases.items():
                expected_noise = np.diag(axes @ noise.mean(0) @ axes.T)
                if expected_noise.min() < -1e-7 * max(1., float(expected_noise.max())):
                    raise AssertionError("Materially negative expected conditional variance")
                empirical_error = np.square((oracle - label) @ axes.T).mean(0)
                basis_results[name] = {
                    "expected_privileged_conditional_noise": pc_summary(expected_noise),
                    "empirical_oracle_label_mse": pc_summary(empirical_error)}
            cells.append({"data_seed": d, "source": source, "alpha": alpha,
                          "n_test": len(data), "window_seed": d + 2,
                          "source_seed": d + 900, "time_seed": d + 901,
                          "geometry_sha256": file_hash(geometry_path),
                          "array_sha256": {name: array_hash(value) for name, value in (
                              ("x", x), ("velocity", label), ("time", time),
                              ("history", data.context), ("raw_history", data.raw_context),
                              ("oracle_velocity", oracle))},
                          "bases": basis_results})
            arrays[d, source] = dict(x=x, label=label, time=time, history=data.context,
                                     oracle=oracle, bases=bases)
            print("reference", d, source, "expected noise",
                  basis_results["source_velocity_pca"]["expected_privileged_conditional_noise"]["overall_mean"], flush=True)
    rows = []
    for r, path in records:
        d, run = r["data_seed"], r["run"]
        samples = args.input / f"dataset_{d}/samples/{run}.npz"
        checkpoint = args.input / f"dataset_{d}/checkpoints/{run}.pt"
        if file_hash(samples) != r["samples_sha256"] or file_hash(checkpoint) != r["checkpoint_sha256"]:
            raise ValueError("Run artifacts do not match recorded hashes: " + run)
        cell = arrays[d, r["source"]]
        with np.load(samples) as saved:
            for key, reference in (("x", "x"), ("velocity", "label"),
                                   ("time", "time"), ("history", "history")):
                np.testing.assert_allclose(saved[key], cell[reference], rtol=1e-12, atol=1e-12,
                                           err_msg="Reference/test alignment " + run + ":" + key)
            prediction = saved["prediction"]
        bases = {name: error_decomposition(prediction, cell["oracle"], cell["label"], axes)
                 for name, axes in cell["bases"].items()}
        for name, metrics_key in (("source_velocity_pca", "metrics"),
                                  ("fixed_target_pca", "fixed_target_basis_metrics")):
            np.testing.assert_allclose(bases[name]["model_label_mse"]["per_pc"],
                                       r[metrics_key]["pc_mse"], atol=1e-10, rtol=1e-10)
        rows.append({"data_seed": d, "run": run, "source": r["source"], "alpha": r["alpha"],
                     "head": r["head"], "seed": r["seed"], "n_test": len(prediction),
                     "run_record_sha256": file_hash(path), "samples_sha256": r["samples_sha256"],
                     "checkpoint_sha256": r["checkpoint_sha256"], "bases": bases})
    source_files = ("scripts/analyze_prior_oracle.py", "fm_stress/data.py", "fm_stress/prior_mixture.py")
    result = {"status": "partial" if pending else "complete", "expected": len(expected_ids),
              "completed": len(rows), "missing": pending, "reference_cell_count": len(cells),
              "fingerprint": manifest["fingerprint"],
              "source_sha256": {name: file_hash(ROOT / name) for name in source_files},
              "cells": cells, "rows": rows,
              "scope": "Analytic privileged GP reference on all aligned held-out velocity pairs; no generation, retraining, hyperparameter choice or checkpoint selection. Reuses existing analytic oracle functions, not an independent numerical audit.",
              "conditioning": "Reference sees x_t, flow time, FULL RAW HISTORY and its location/scale. Learned heads see x_t, flow time and a tiny encoder of normalized history. This stronger-information reference is not an attainable Bayes oracle for those models.",
              "decomposition": "Empirically MSE(pred,label)=MSE(pred,oracle)+MSE(oracle,label)+mean(2*(pred-oracle)*(oracle-label)) coordinatewise. Signed sample cross term is retained. Expected conditional noise averages analytic covariance over the observed histories/times; it is not forcibly equal to the realized oracle label error.",
              "interpretation": "Priors change both regression geometry and conditional label noise. Compare methods within each source, and use fixed target axes for cross-source orientation. Do not equate model MSE minus the analytic floor with an unbiased finite-sample estimation error; compare the directly computed model-oracle discrepancy and keep the cross term. Original raw metrics and decision rule remain unchanged."}
    save_json(args.output, result)
    print(result["status"], len(rows), "/", len(expected_ids), flush=True)


if __name__ == "__main__":
    main()
