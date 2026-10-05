"""Bounded lower-rate warm-restart sensitivity, separate from primary results.

Starts each GP-source head from its validation-selected primary checkpoint.
Adam moments are reset because primary checkpoints contain model weights only.
The initial checkpoint remains eligible; all further selection uses validation
labels only. This is an optimization sensitivity, not an extension or replacement
of the original equal-budget four-cell experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import source_covariance
from .experiment import (config_hash, fixed_pairs, generate, get_data, load_config,
                         predict, save_json, tensor)
from .metrics import PCA, evaluate
from .models import optimizer_groups
from .robustness import restore_checkpoint, s4_stability


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def warm_restart(config, results, output, kind, seed, device, additional_steps=3000,
                 learning_rate=3e-4, eval_every=500, n_paths=None):
    if Path(results).resolve() == Path(output).resolve():
        raise ValueError("Sensitivity output must differ from the primary results directory")
    gp, data = get_data(config)
    name = f"gp_{kind}_seed{seed}"
    source_checkpoint = Path(results) / "checkpoints" / f"{name}.pt"
    source_run = Path(results) / "runs" / f"{name}.json"
    options = {"kind": kind, "seed": seed, "source": "gp", "additional_steps": additional_steps,
               "learning_rate": learning_rate, "eval_every": eval_every,
               "n_paths": config["n_paths"] if n_paths is None else n_paths,
               "training_rng_seed": 30000 + seed, "config_sha256": config_hash(config),
               "initial_checkpoint_sha256": file_sha256(source_checkpoint)}
    out = Path(output)
    record_path = out / "runs" / f"{name}.json"
    if record_path.exists():
        existing = json.loads(record_path.read_text())
        if existing.get("options") != options:
            raise RuntimeError(f"Refusing to overwrite different sensitivity run {record_path}")
        print("SKIP", name, flush=True)
        return existing
    baseline = json.loads(source_run.read_text())
    if baseline["config_sha256"] != config_hash(config):
        raise ValueError("Baseline metrics/configuration mismatch")
    model, initial = restore_checkpoint(source_checkpoint, config, device)
    p = initial["pca"]
    pca = PCA(np.asarray(p["mean"]), np.asarray(p["components"]),
              np.asarray(p["eigenvalues"]), float(p["floor"]))
    if initial["source"] != "gp" or initial["kind"] != kind or initial["seed"] != seed:
        raise ValueError("Checkpoint identity does not match requested cell")
    started = time.perf_counter()
    target = tensor(data["train"].target, device)
    history = tensor(data["train"].context, device)
    chol = tensor(np.linalg.cholesky(source_covariance(gp, "gp")), device)
    vx, vu, vt, _ = fixed_pairs(config, gp, data["val"], "gp", "val")
    vx, vu, vt, vh = [tensor(a, device) for a in (vx, vu, vt, data["val"].context)]
    optimizer = torch.optim.AdamW(optimizer_groups(model, learning_rate, config["weight_decay"]))

    def validation():
        model.eval()
        with torch.no_grad():
            prediction = predict(model, vx, vt, vh)
            objective = (model.to_coordinates(prediction-vu)).square().mean().item()
            raw = (prediction-vu).square().mean().item()
        if not np.isfinite(objective):
            raise FloatingPointError(f"Non-finite validation loss for {name}")
        return objective, raw

    initial_objective, initial_raw = validation()
    # Accommodate tiny device/reduction roundoff, but catch wrong checkpoint/data.
    if not np.isclose(initial_objective, baseline["best_validation_objective"], rtol=1e-4, atol=1e-5):
        raise ValueError("Restored initial validation objective differs from recorded primary best")
    best, best_additional_step = initial_objective, 0
    best_raw = initial_raw
    curve = [{"additional_step": 0, "validation_objective": initial_objective,
              "validation_raw_mse": initial_raw, "elapsed_seconds": 0.0}]
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    selected_checkpoint = checkpoint_dir / f"{name}.pt"

    def save_checkpoint(additional_step):
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": config, "kind": kind, "source": "gp", "seed": seed,
                    "step": initial["step"] + additional_step, "pca": initial["pca"],
                    "sensitivity": "lower-rate warm restart; Adam reset; initial best included",
                    "initial_checkpoint_step": initial["step"],
                    "additional_step": additional_step, "options": options}, selected_checkpoint)

    save_checkpoint(0)
    torch.manual_seed(options["training_rng_seed"])
    if device.type == "mps":
        torch.mps.manual_seed(options["training_rng_seed"])
    print("START WARM RESTART", name, "from", initial["step"], "val", best,
          "additional", additional_steps, "lr", learning_rate, flush=True)
    for step in range(1, additional_steps + 1):
        model.train()
        indices = torch.randint(len(target), (config["batch_size"],), device=device)
        y, h = target[indices], history[indices]
        eps = torch.randn(len(indices), gp.horizon, device=device) @ chol.T
        t = torch.rand(len(indices), device=device)
        x, u = (1-t[:, None])*eps + t[:, None]*y, y-eps
        optimizer.zero_grad(set_to_none=True)
        pred = model.forward_coordinates(x, t, h)
        loss = (pred-model.to_coordinates(u)).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
        optimizer.step()
        if step % eval_every == 0 or step == additional_steps:
            objective, raw = validation()
            row = {"additional_step": step, "train_loss": loss.item(),
                   "validation_objective": objective, "validation_raw_mse": raw,
                   "elapsed_seconds": time.perf_counter()-started}
            curve.append(row)
            if objective < best:
                best, best_additional_step = objective, step
                best_raw = raw
                save_checkpoint(step)
            save_json(out / "training" / f"{name}.json", curve)
            print(name, "extra", step, "val", round(objective, 6), "raw", round(raw, 6), flush=True)

    selected = torch.load(selected_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["state_dict"])
    model.eval()
    # Test evaluation occurs only after the additional validation selection.
    tx, tu, tt, teps = fixed_pairs(config, gp, data["test"], "gp", "test")
    prediction = predict(model, tensor(tx, device), tensor(tt, device),
                         tensor(data["test"].context, device)).cpu().numpy()
    n = min(options["n_paths"], len(tx))
    generated = generate(model, tensor(teps[:n], device),
                         tensor(data["test"].context[:n], device), config["euler_steps"])
    metrics = evaluate(prediction, tu, generated, data["test"].target[:n], pca)
    record = {"run": name, "source": "gp", "head": kind, "seed": seed, "options": options,
              "status": "complete", "device": str(device), "torch_version": torch.__version__,
              "protocol": "Separate sensitivity: reset Adam, lower LR, same additional budget/RNG across heads, select on validation including initial checkpoint; primary results unchanged.",
              "config_sha256": config_hash(config),
              "source_checkpoint": str(source_checkpoint), "source_metrics": str(source_run),
              "initial_checkpoint_step": int(initial["step"]),
              "primary_training_budget": config["steps"],
              "total_training_budget": config["steps"] + additional_steps,
              "initial_validation_objective": initial_objective,
              "initial_validation_raw_mse": initial_raw,
              "best_additional_step": best_additional_step,
              "best_validation_objective": best,
              "best_validation_raw_mse": best_raw,
              "validation_improvement_fraction": (initial_objective-best)/initial_objective,
              "selected_checkpoint": str(selected_checkpoint),
              "baseline_metrics": baseline["metrics"], "metrics": metrics,
              "baseline_roughness_same_path_count": n == min(config["n_paths"], len(tx)),
              "curve": curve, "elapsed_seconds": time.perf_counter()-started,
              "s4_internal_stability": s4_stability(model)}
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(samples_dir / f"{name}.npz", generated=generated,
                        target=data["test"].target[:n], history=data["test"].context[:n],
                        prediction=prediction, velocity=tu, time=tt, source=teps[:n])
    save_json(record_path, record)
    print("DONE WARM RESTART", name, "validation improvement", round(100*record["validation_improvement_fraction"], 3),
          "percent; roughness ratio", round(metrics["success"]["roughness_ratio"], 4), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--heads", default="mlp,s4,whitened_mlp")
    parser.add_argument("--additional-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--n-paths", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    heads, seeds = args.heads.split(","), [int(s) for s in args.seeds.split(",")]
    if not set(heads) <= {"mlp", "s4", "whitened_mlp"}:
        parser.error("Unknown head")
    if args.additional_steps < 1 or args.eval_every < 1 or args.learning_rate <= 0:
        parser.error("Additional budget, evaluation interval and learning rate must be positive")
    if args.n_paths is not None and args.n_paths < 1:
        parser.error("n-paths must be positive")
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not exposed to this process")
    torch.set_num_threads(4)
    config = load_config(args.config)
    output = args.output or str(Path(args.results) / "optimization_sensitivity")
    records = []
    for seed in seeds:
        for head in heads:
            records.append(warm_restart(config, args.results, output, head, seed,
                                        torch.device(device_name), args.additional_steps,
                                        args.learning_rate, args.eval_every, args.n_paths))
    # Keep seed0 in the summary when a later command adds seeds1,2.
    records = [json.loads(path.read_text()) for path in sorted((Path(output)/"runs").glob("*.json"))]
    records = [r for r in records if r["config_sha256"] == config_hash(config)
               and r["options"]["additional_steps"] == args.additional_steps
               and r["options"]["learning_rate"] == args.learning_rate]
    save_json(Path(output) / "summary.json", {
        "protocol": "Post-primary lower-rate warm-restart sensitivity, not primary model selection.",
        "config_sha256": config_hash(config), "seeds": sorted({r["seed"] for r in records}),
        "heads": sorted({r["head"] for r in records}),
        "runs": [{"run": r["run"], "validation_improvement_fraction": r["validation_improvement_fraction"],
                  "best_additional_step": r["best_additional_step"], "metrics": r["metrics"]}
                 for r in records]})


if __name__ == "__main__":
    main()
