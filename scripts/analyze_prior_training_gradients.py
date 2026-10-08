"""Postprocess selected checkpoints: raw versus actual training-loss gradients.

Reuses the frozen gradient helper on the same 64 validation probes. This is an
additional diagnostic, not independent validation and not model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.data import GPConfig, sample_dataset
from fm_stress.experiment import load_config, save_json
from fm_stress.metrics import PCA
from fm_stress.prior_mixture import (array_hash, file_hash, fixed_pairs, identities,
                                     make_model, objective, source_name)
from fm_stress.prior_mixture_diagnostics import gradient_diagnostics


class TrainingCoordinates(torch.nn.Module):
    """Expose the existing balanced head's output in weighted PCA coordinates."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, history):
        return self.model.coordinates(self.model(x, t, history))


def ratio_summary(diagnostic):
    a = diagnostic["groups"]["pc1"]["gradient_norm"]
    b = diagnostic["groups"]["residual"]["gradient_norm"]
    return {"pc1_over_residual_gradient_norm": a / b if b > 0 else None,
            "pc1_residual_cosine": diagnostic["gradient_cosines"]["pc1_vs_residual"]}


def actual_training_diagnostics(model, x, t, history, velocity, pca):
    """Return helper output with explicit loss/coordinate semantics corrected."""
    if model.balanced:
        f = velocity.shape[1]
        identity = PCA(np.zeros(f), np.eye(f), np.ones(f), 1e-8)
        target = model.coordinates(velocity).detach()
        value = gradient_diagnostics(TrainingCoordinates(model), x, t, history, target, identity)
        value["coordinate_system"] = "source_specific_velocity_PCA_divided_by_train_scales"
        value["loss_definition"] = "mean(((prediction - velocity) @ PCA_rows.T / scales)^2)"
    else:
        value = gradient_diagnostics(model, x, t, history, velocity, pca)
        value["loss_definition"] = "mean((prediction - velocity)^2)"
    # The frozen helper's generic raw_* keys refer to its passed coordinates;
    # rename them here so downstream reporting cannot mistake weighted for raw.
    value["objective_mse"] = value.pop("raw_total_mse")
    value["objective_total_gradient_norm"] = value.pop("raw_total_gradient_norm")
    for group in value["groups"].values():
        group["objective_mse_contribution"] = group.pop("raw_mse_contribution")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/prior-mixture")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/diagnostics/prior_mixture_training_gradients.json")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = load_config(args.input / "config.json")
    manifest = load_config(args.input / "manifest.json")
    for name, expected in manifest["source_hashes"].items():
        if file_hash(ROOT / name) != expected:
            raise ValueError("Frozen source hash changed: " + name)
    source_files = ["scripts/analyze_prior_training_gradients.py",
                    "fm_stress/prior_mixture.py", "fm_stress/prior_mixture_diagnostics.py"]
    provenance = {name: file_hash(ROOT / name) for name in source_files}
    cache = {}
    if args.output.exists():
        old = load_config(args.output)
        if old.get("source_sha256") == provenance and old.get("fingerprint") == manifest["fingerprint"]:
            cache = {(r["data_seed"], r["run"]): r for r in old["rows"]}
    expected_ids = identities(cfg)
    records = []
    missing = []
    for d, alpha, head, seed in expected_ids:
        name = f"{source_name(alpha)}_{head}_seed{seed}"
        path = args.input / f"dataset_{d}/runs/{name}.json"
        if not path.exists():
            missing.append({"data_seed": d, "run": name})
            continue
        r = load_config(path)
        if r["status"] != "complete" or r["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Incomplete or incompatible run: " + str(path))
        records.append((r, path))
    if missing and not args.allow_partial:
        raise ValueError(f"Require all {len(expected_ids)} runs; missing {len(missing)}")
    data = {}
    rows = []
    for r, path in records:
        d, name = r["data_seed"], r["run"]
        checkpoint = args.input / f"dataset_{d}/checkpoints/{name}.pt"
        checkpoint_sha = file_hash(checkpoint)
        if checkpoint_sha != r["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch: " + str(checkpoint))
        if (d, name) in cache and cache[d, name]["checkpoint_sha256"] == checkpoint_sha:
            rows.append(cache[d, name])
            continue
        if d not in data:
            data[d] = sample_dataset(GPConfig(**cfg["data"]), cfg["n_val"], d + 1)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        p = ck["pca"]
        pca = PCA(np.array(p["mean"]), np.array(p["components"]),
                  np.array(p["eigenvalues"]), p["floor"])
        with torch.random.fork_rng(devices=[]):
            model = make_model(cfg, r["head"], pca, r["seed"], torch.device("cpu"))
        model.load_state_dict(ck["state_dict"])
        model.eval()
        x, u, t, _ = fixed_pairs(cfg, data[d], r["alpha"], d, "val")
        n = cfg["gradient_probe_size"]
        x, u, t, h = [torch.tensor(a[:n], dtype=torch.float32)
                      for a in (x, u, t, data[d].context)]
        rng = torch.get_rng_state().clone()
        state_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        raw = gradient_diagnostics(model, x, t, h, u, pca)
        actual = actual_training_diagnostics(model, x, t, h, u, pca)
        if not torch.equal(rng, torch.get_rng_state()):
            raise AssertionError("Gradient diagnostics consumed RNG")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise AssertionError("Gradient diagnostics mutated .grad")
        if any(not torch.equal(value, model.state_dict()[name]) for name, value in state_before.items()):
            raise AssertionError("Gradient diagnostics mutated model state")
        with torch.no_grad():
            optimized_loss = objective(model, model(x, t, h), u).item()
        if not np.isclose(optimized_loss, actual["objective_mse"], atol=2e-4, rtol=2e-4):
            raise AssertionError("Diagnostic objective differs from optimized objective")
        selected = [g for g in r["gradient_diagnostics"] if g.get("selected_checkpoint")]
        if len(selected) != 1 or selected[0]["step"] != ck["step"]:
            raise AssertionError("Selected checkpoint diagnostic missing or ambiguous")
        row = {"data_seed": d, "run": name, "alpha": r["alpha"], "head": r["head"],
               "seed": r["seed"], "selected_step": ck["step"],
               "checkpoint_sha256": checkpoint_sha, "run_record_sha256": file_hash(path),
               "balanced_objective": bool(model.balanced),
               "probe": {"split": "validation", "count": n, "window_seed": d + 1,
                         "source_seed": d + 800, "time_seed": d + 801,
                         "selection": "first gradient_probe_size rows of complete validation draws",
                         "float32_arrays_sha256": {key: array_hash(value.numpy())
                                                   for key, value in (("x", x), ("u", u), ("t", t), ("history", h))}},
               "raw": raw, "training_objective": actual,
               "raw_summary": ratio_summary(raw), "training_objective_summary": ratio_summary(actual),
               "optimized_probe_loss": optimized_loss,
               "objective_reconstruction_absolute_error": abs(optimized_loss - actual["objective_mse"]),
               "recorded_mps_raw_total_gradient_norm": selected[0]["raw_total_gradient_norm"],
               "cpu_raw_total_gradient_norm": raw["raw_total_gradient_norm"],
               "rng_grad_and_model_state_preserved": True}
        rows.append(row)
        print(d, name, "raw", round(row["raw_summary"]["pc1_over_residual_gradient_norm"] or 0, 4),
              "optimized", round(row["training_objective_summary"]["pc1_over_residual_gradient_norm"] or 0, 4), flush=True)
    result = {"status": "partial" if missing else "complete", "device": "cpu",
              "expected": len(expected_ids), "completed": len(rows), "missing": missing,
              "fingerprint": manifest["fingerprint"], "source_sha256": provenance,
              "rows": rows,
              "scope": "Selected checkpoints only; same fixed validation probes as training runner. Reuses frozen model/gradient implementation; extra diagnostic, not independent numerical validation.",
              "interpretation": "Raw and actual objective refer to gradients in the same model parameterization. Compare their within-model group balance. Cross-head absolute norms are not invariant; balanced weighting changes both group scales and potentially gradient conflict. Neither gradient imbalance nor cosine alone establishes a causal explanation.",
              "groups": "PC1, PC2, residual PCs3:F share denominator batch_size*horizon; residual is summed over its F-2 directions. Balanced objective uses train-fitted source-specific velocity PCA scales, including its regularizing floor."}
    save_json(args.output, result)
    print(result["status"], len(rows), "/", len(expected_ids), flush=True)


if __name__ == "__main__":
    main()
