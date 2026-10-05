"""Prospectively fixed, fresh-context conditional forecast evaluation.

All comparisons use shared contexts and source ensembles. No training, tuning or
checkpoint selection occurs here. See docs/ensemble_evaluation_protocol.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import pdist
import torch

from .data import GPConfig, sample_dataset, sample_source, sample_conditional_target
from .experiment import config_hash, generate, load_config, save_json, tensor
from .metrics import paired_bootstrap_difference
from .robustness import restore_checkpoint


def _validated(samples, target):
    x, y = np.asarray(samples, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if x.ndim != 3 or y.shape != (x.shape[0], x.shape[2]) or x.shape[1] < 2:
        raise ValueError("Expected finite samples [H,S,F], S>=2, and target [H,F]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite ensemble or target")
    return x, y


def fair_crps(samples, target):
    """Per-history marginal CRPS, unbiased off-diagonal ensemble estimator."""
    x, y = _validated(samples, target)
    s = x.shape[1]
    weights = 2 * np.arange(1, s + 1) - s - 1
    # sum_{i<j}|x_i-x_j| = sum_j (2j-S-1) x_(j).
    unordered_sum = np.sum(np.sort(x, axis=1) * weights[None, :, None], axis=1)
    return (np.abs(x-y[:, None]).mean(axis=1) - unordered_sum/(s*(s-1))).mean(axis=1)


def fair_energy_score(samples, target):
    """Per-history Euclidean path energy score with off-diagonal correction."""
    x, y = _validated(samples, target)
    s = x.shape[1]
    return np.linalg.norm(x-y[:, None], axis=2).mean(axis=1) - np.array([
        pdist(ensemble, metric="euclidean").sum()/(s*(s-1)) for ensemble in x])


def interval_metrics(samples, target, level):
    """Marginal central coverage/width, linear (type-7) empirical quantiles."""
    x, y = _validated(samples, target)
    if not 0 < level < 1:
        raise ValueError("Interval level must lie in (0,1)")
    alpha = (1-level)/2
    lo, hi = np.quantile(x, [alpha, 1-alpha], axis=1, method="linear")
    return ((y >= lo) & (y <= hi)).mean(axis=1), (hi-lo).mean(axis=1)


def context_scores(samples, target):
    x, y = _validated(samples, target)
    if x.shape[2] < 2:
        raise ValueError("Roughness requires at least two future coordinates")
    metrics = {
        "fair_crps": fair_crps(x, y),
        "fair_energy_score": fair_energy_score(x, y),
        "mean_forecast_mse": np.square(x.mean(axis=1)-y).mean(axis=1),
        "ensemble_variance": x.var(axis=1, ddof=1).mean(axis=1),
        "generated_roughness": np.abs(np.diff(x, axis=2)).mean(axis=(1, 2)),
        "target_roughness": np.abs(np.diff(y, axis=1)).mean(axis=1),
    }
    for level in (.8, .9):
        coverage, width = interval_metrics(x, y, level)
        metrics[f"coverage_{int(level*100)}"] = coverage
        metrics[f"width_{int(level*100)}"] = width
    return metrics


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_inputs(gp, cfg):
    ds = sample_dataset(gp, cfg["n_histories"], cfg["data_seed"])
    sources = sample_source(gp, cfg["n_histories"]*cfg["ensemble_size"], "gp", cfg["source_seed"])
    return ds, sources.reshape(cfg["n_histories"], cfg["ensemble_size"], gp.horizon)


def aggregate(records, cfg, fingerprint):
    groups = {h: sorted([r for r in records if r["head"] == h], key=lambda r:r["seed"])
              for h in cfg["heads"]}
    expected = {(head, seed) for head in cfg["heads"] for seed in cfg["seeds"]}
    observed = {(r["head"], r["seed"]) for r in records}
    complete = expected == observed and len(records) == len(expected)
    summary = {"complete": complete, "fingerprint": fingerprint,
               "expected_runs": len(expected), "observed_runs": len(records),
               "missing": sorted(expected-observed), "heads": {}, "contrasts": []}
    for head, rows in groups.items():
        if not rows:
            continue
        metrics = rows[0]["means"].keys()
        summary["heads"][head] = {"n_seeds": len(rows), "metrics": {
            metric: {"mean": float(np.mean([r["means"][metric] for r in rows])),
                     "seed_sd": float(np.std([r["means"][metric] for r in rows], ddof=1)) if len(rows)>1 else None}
            for metric in metrics}}
    # Do not summarize a selected subset as the prospectively fixed comparison.
    if complete:
        for a, b in (("s4", "mlp"), ("whitened_mlp", "mlp")):
            for metric in records[0]["per_context"]:
                av = np.mean([r["per_context"][metric] for r in groups[a]], axis=0)
                bv = np.mean([r["per_context"][metric] for r in groups[b]], axis=0)
                ci = paired_bootstrap_difference(av, bv, seed=cfg["bootstrap_seed"],
                                                 repetitions=cfg["bootstrap_repetitions"])
                summary["contrasts"].append({"a": a, "b": b, "metric": metric,
                    "unit": "paired fresh histories; scores averaged over fixed training seeds", **ci})
    return summary


def solver_sensitivity(cfg, training, ds, sources, checkpoint_paths, out, device, fingerprint):
    options = cfg.get("solver_sensitivity")
    if not options:
        return
    directory = out / "solver_sensitivity"
    directory.mkdir(exist_ok=True)
    n, seed, k = options["n_histories"], options["seed"], options["euler_steps"]
    h = tensor(np.repeat(ds.context[:n], cfg["ensemble_size"], axis=0), device)
    eps = tensor(sources[:n].reshape(-1, sources.shape[-1]), device)
    rows = []
    for head in cfg["heads"]:
        name = f"gp_{head}_seed{seed}"
        record_path = directory / f"{name}.json"
        if record_path.exists():
            record = load_config(record_path)
            if record["fingerprint"] != fingerprint:
                raise RuntimeError("Solver sensitivity fingerprint mismatch")
            rows.append(record)
            continue
        started = time.perf_counter()
        model, _ = restore_checkpoint(checkpoint_paths[name], training, device)
        print("SOLVER START", name, "Euler", k, "histories", n, flush=True)
        new = generate(model, eps, h, k, batch_size=cfg["batch_size"]).reshape(n, cfg["ensemble_size"], -1)
        old = np.load(out/"samples"/f"{name}.npz")["generated"][:n]
        old_scores, new_scores = context_scores(old, ds.target[:n]), context_scores(new, ds.target[:n])
        record = {"name": name, "head": head, "seed": seed, "n_histories": n,
                  "euler_steps": k, "baseline_euler_steps": cfg["euler_steps"],
                  "fingerprint": fingerprint, "device": str(device),
                  "seconds": time.perf_counter()-started,
                  "paired_endpoint_rmse": float(np.sqrt(np.square(new.astype(float)-old).mean())),
                  "baseline_means": {key:float(v.mean()) for key,v in old_scores.items()},
                  "new_means": {key:float(v.mean()) for key,v in new_scores.items()},
                  "paired_context_metric_differences": {key:(new_scores[key]-old_scores[key]).tolist() for key in new_scores}}
        np.savez_compressed(directory/f"{name}.npz", generated=new.astype(np.float32))
        save_json(record_path, record)
        rows.append(record)
        print("SOLVER DONE", name, "endpoint RMSE", record["paired_endpoint_rmse"], flush=True)
        del model
    save_json(directory/"summary.json", {"fingerprint": fingerprint, "rows": rows,
        "scope": "Predetermined first 32 fresh histories, seed 0, common ensemble sources; diagnostic only"})


def run(cfg, device, head_filter=None, seed_filter=None):
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; no silent CPU fallback")
    torch.set_num_threads(4)
    training = load_config(cfg["training_config"])
    gp = GPConfig(**training["data"])
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = {f"gp_{h}_seed{s}": Path(cfg["checkpoint_directory"])/f"gp_{h}_seed{s}.pt"
                        for h in cfg["heads"] for s in cfg["seeds"]}
    module_files = ["data.py", "metrics.py", "models.py", "experiment.py", "robustness.py", "ensemble_evaluation.py"]
    manifest = {"config": cfg, "training_config_sha256": config_hash(training),
                "checkpoint_sha256": {k:file_hash(p) for k,p in checkpoint_paths.items()},
                "source_sha256": {name:file_hash(Path(__file__).parent/name) for name in module_files}}
    fingerprint = config_hash(manifest)
    manifest["fingerprint"] = fingerprint
    if (out/"manifest.json").exists():
        old = load_config(out/"manifest.json")
        if old["fingerprint"] != fingerprint:
            raise RuntimeError("Refusing to mix changed protocol/checkpoints/source in the same output directory")
    save_json(out/"manifest.json", manifest)
    save_json(out/"config.json", cfg)
    ds, sources = shared_inputs(gp, cfg)
    np.savez_compressed(out/"common_inputs.npz", history=ds.context, target=ds.target,
                        raw_history=ds.raw_context, source=sources)
    repeated_history = np.repeat(ds.context, cfg["ensemble_size"], axis=0)
    flat_sources = sources.reshape(-1, gp.horizon)
    source_tensor, history_tensor = tensor(flat_sources, device), tensor(repeated_history, device)
    (out/"runs").mkdir(exist_ok=True)
    (out/"samples").mkdir(exist_ok=True)
    for head in cfg["heads"]:
        if head_filter is not None and head != head_filter:
            continue
        for seed in cfg["seeds"]:
            if seed_filter is not None and seed != seed_filter:
                continue
            name = f"gp_{head}_seed{seed}"
            record_path = out/"runs"/f"{name}.json"
            if record_path.exists():
                old = load_config(record_path)
                if old["fingerprint"] != fingerprint or not (out/"samples"/f"{name}.npz").exists():
                    raise RuntimeError(f"Inconsistent completed record {name}")
                print("SKIP", name, flush=True)
                continue
            started = time.perf_counter()
            model, checkpoint = restore_checkpoint(checkpoint_paths[name], training, device)
            print("START", name, "histories", len(ds), "ensemble", cfg["ensemble_size"],
                  "Euler", cfg["euler_steps"], "device", device, flush=True)
            generated = generate(model, source_tensor, history_tensor, cfg["euler_steps"],
                                 batch_size=cfg["batch_size"]).reshape(sources.shape)
            metrics = context_scores(generated, ds.target)
            record = {"name": name, "head": head, "seed": seed, "device": str(device),
                      "torch": torch.__version__, "checkpoint_step": checkpoint["step"],
                      "fingerprint": fingerprint, "seconds": time.perf_counter()-started,
                      "means": {k:float(v.mean()) for k,v in metrics.items()},
                      "per_context": {k:v.tolist() for k,v in metrics.items()}}
            np.savez_compressed(out/"samples"/f"{name}.npz", generated=generated.astype(np.float32))
            save_json(record_path, record)
            print("DONE", name, "CRPS", record["means"]["fair_crps"], "energy",
                  record["means"]["fair_energy_score"], "seconds", record["seconds"], flush=True)
            del model
    if cfg.get("include_privileged_reference", False) and not (out/"privileged_reference.json").exists():
        reference = sample_conditional_target(gp, ds, samples=cfg["ensemble_size"], seed=cfg["reference_seed"])
        metrics = context_scores(reference, ds.target)
        save_json(out/"privileged_reference.json", {
            "label": "privileged exact-conditional GP reference",
            "information": "raw history, unlike normalized history given to networks",
            "caveat": "Not an attainable network baseline or a finite-ensemble lower bound for proper scores",
            "fingerprint": fingerprint, "means": {k:float(v.mean()) for k,v in metrics.items()},
            "per_context": {k:v.tolist() for k,v in metrics.items()}})
        np.savez_compressed(out/"samples"/"privileged_reference.npz", generated=reference.astype(np.float32))
    records = [load_config(p) for p in sorted((out/"runs").glob("*.json"))]
    if any(r["fingerprint"] != fingerprint for r in records):
        raise RuntimeError("Result fingerprint mismatch")
    summary = aggregate(records, cfg, fingerprint)
    save_json(out/"summary.json", summary)
    if summary["complete"]:
        solver_sensitivity(cfg, training, ds, sources, checkpoint_paths, out, device, fingerprint)
    print("COMPLETE" if summary["complete"] else "PARTIAL", len(records), "evaluations", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ensemble_evaluation.json")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--head", choices=["mlp", "s4", "whitened_mlp"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    run(load_config(args.config), torch.device(args.device), args.head, args.seed)


if __name__ == "__main__":
    main()
