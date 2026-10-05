"""Read-only solver and S4-stability diagnostics for saved checkpoints.

Example:
  python -m fm_stress.robustness --config configs/main.json --results results/main \
      --device mps --seed 0 --n-paths 256 --steps 32,64,128,256

This never trains or selects checkpoints. All step counts use identical test
histories and initial source samples. Diagnostics are supplemental: the fixed
Euler count in the preregistered main experiment is not changed retrospectively.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch
from scipy.special import erf

from .data import conditional_target, source_covariance
from .experiment import (CELLS, CoordinateModel, config_hash, fixed_pairs,
                         generate, get_data, load_config, save_json, tensor)
from .metrics import PCA, roughness
from .models import S4NPLRKernel, create_model


def restore_checkpoint(path, config, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if config_hash(checkpoint["config"]) != config_hash(config):
        raise ValueError(f"Checkpoint configuration differs from requested config: {path}")
    p = checkpoint["pca"]
    pca = PCA(np.asarray(p["mean"]), np.asarray(p["components"]),
              np.asarray(p["eigenvalues"]), float(p["floor"]))
    kind = checkpoint["kind"]
    base = create_model("mlp" if kind == "whitened_mlp" else kind,
                        horizon=config["data"]["horizon"],
                        lookback=config["data"]["lookback"], **config["model"])
    model = CoordinateModel(base, pca if kind == "whitened_mlp" else None)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def s4_stability(model):
    """CPU-float64 inspection of learned continuous/discrete S4 spectra."""
    diagnostics = []
    for name, module in model.named_modules():
        if not isinstance(module, S4NPLRKernel):
            continue
        # Inspect copies; do not change checkpoint or model device/precision.
        ssm = S4NPLRKernel(module.width, module.state_size,
                           directions=module.c.shape[0]).double()
        ssm.load_state_dict({key: value.detach().cpu().double()
                             for key, value in module.state_dict().items()})
        ar = -ssm.log_a_real.exp().numpy()
        omega = ssm.a_imag.numpy()
        n = module.state_size // 2
        eye = np.eye(n)
        a = np.concatenate((np.concatenate((ar[..., None] * eye, -omega[..., None] * eye), axis=-1),
                            np.concatenate((omega[..., None] * eye, ar[..., None] * eye), axis=-1)), axis=-2)
        p = ssm.p.detach().numpy()
        a -= 2 * p[..., :, None] * p[..., None, :]
        ab, _ = ssm.discretize()
        eigenvalues = np.linalg.eigvals(a)
        discrete_eigenvalues = np.linalg.eigvals(ab.numpy())
        max_real = float(eigenvalues.real.max())
        radius = float(np.abs(discrete_eigenvalues).max())
        diagnostics.append({"layer": name, "max_continuous_real_eigenvalue": max_real,
                            "max_discrete_spectral_radius": radius,
                            "stable": bool(max_real < 0 and radius <= 1 + 1e-9),
                            "dt_min": float(ssm.log_dt.exp().min()),
                            "dt_max": float(ssm.log_dt.exp().max()),
                            "kernel_abs_max": float(ssm(16).abs().max())})
    return diagnostics


def learned_solver_check(config, results, source, kind, seed, steps, n_paths, device):
    gp, data = get_data(config)
    name = f"{source}_{kind}_seed{seed}"
    path = Path(results) / "checkpoints" / f"{name}.pt"
    model, checkpoint = restore_checkpoint(path, config, device)
    _, _, _, eps = fixed_pairs(config, gp, data["test"], source, "test")
    n = min(n_paths, len(eps))
    target = data["test"].target[:n]
    source_tensor = tensor(eps[:n], device)
    histories = tensor(data["test"].context[:n], device)
    target_r = roughness(target)
    paths, rows = {}, []
    for k in steps:
        started = time.perf_counter()
        paths[k] = generate(model, source_tensor, histories, k)
        generated_r = roughness(paths[k])
        row = {"euler_steps": k, "generated_roughness": generated_r,
               "roughness_to_target": generated_r / target_r,
               "endpoint_mean": float(paths[k].mean()), "endpoint_std": float(paths[k].std()),
               "max_absolute_value": float(np.abs(paths[k]).max()),
               "seconds": time.perf_counter() - started}
        rows.append(row)
        print(name, "Euler", k, "roughness ratio", round(row["roughness_to_target"], 5), flush=True)
    largest = max(steps)
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    for i, row in enumerate(rows):
        k = row["euler_steps"]
        rmse = float(np.sqrt(np.mean((paths[k] - paths[largest]) ** 2)))
        row["paired_endpoint_rmse_to_largest_k"] = rmse
        row["paired_endpoint_rmse_over_target_rms"] = rmse / target_rms
        row["roughness_relative_change_to_largest_k"] = float(roughness(paths[k]) / roughness(paths[largest]) - 1)
        if i:
            previous = rows[i - 1]["euler_steps"]
            row["paired_endpoint_rmse_to_previous_k"] = float(np.sqrt(np.mean((paths[k] - paths[previous]) ** 2)))
    result = {"checkpoint": str(path), "checkpoint_step": int(checkpoint["step"]),
              "source": source, "kind": kind, "seed": seed, "n_paths": n,
              "device": str(device), "target_roughness": target_r,
              "stability_scope": "S4 spectra concern internal linear SSMs only; they do not prove stability of the complete nonlinear velocity ODE.",
              "rows": rows, "s4_stability": s4_stability(model)}
    return name, result


def oracle_euler_transport(covariance, source_cov, steps):
    """Exact covariance of explicit-Euler oracle samples (float64 CPU).

    For x_t = t mu + T_t epsilon, dT/dt = A_t T, with
    A_t = [t C_y - (1-t) C_eps] [t² C_y +(1-t)² C_eps]^-1.
    The mean t*mu is integrated exactly by Euler. We propagate the full linear
    transfer rather than estimating conditional covariance from one sample.
    """
    n, f, _ = covariance.shape
    transport = np.broadcast_to(np.eye(f), (n, f, f)).copy()
    for j in range(steps):
        t = j / steps
        sx = t*t*covariance + (1-t)**2*source_cov
        ux = t*covariance - (1-t)*source_cov
        a = np.linalg.solve(sx, ux.swapaxes(-1, -2)).swapaxes(-1, -2)
        transport = transport + (a @ transport) / steps
    endpoint_cov = transport @ source_cov @ transport.swapaxes(-1, -2)
    return transport, (endpoint_cov + endpoint_cov.swapaxes(-1, -2)) / 2


def covariance_diagnostics(actual, target):
    f = target.shape[-1]
    chol = np.linalg.cholesky(target)
    left = np.linalg.solve(chol, actual)
    whitened = np.linalg.solve(chol, left.swapaxes(-1, -2)).swapaxes(-1, -2)
    whitened = (whitened + whitened.swapaxes(-1, -2)) / 2
    ratios = np.linalg.eigvalsh(whitened)
    relative_frobenius = np.linalg.norm(actual-target, axis=(-2,-1)) / np.linalg.norm(target, axis=(-2,-1))
    trace_ratio = np.trace(actual, axis1=-2, axis2=-1) / np.trace(target, axis1=-2, axis2=-1)
    return {"conditional_trace_ratio_mean": float(trace_ratio.mean()),
            "conditional_trace_ratio_min": float(trace_ratio.min()),
            "relative_conditional_covariance_error_mean": float(relative_frobenius.mean()),
            "relative_conditional_covariance_error_max": float(relative_frobenius.max()),
            "whitened_covariance_eigenvalue_min": float(ratios.min()),
            "whitened_covariance_eigenvalue_median": float(np.median(ratios)),
            "whitened_covariance_eigenvalue_max": float(ratios.max()),
            "fraction_conditional_directions_below_80pct_variance": float(np.mean(ratios < .8)),
            "fraction_conditional_directions_above_120pct_variance": float(np.mean(ratios > 1.2))}


def gaussian_expected_roughness(mean, covariance):
    """Exact E mean|x[i+1]-x[i]| for history-conditional Gaussians."""
    delta = np.diff(mean, axis=-1)
    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    variance = diagonal[:, 1:] + diagonal[:, :-1] - 2*np.diagonal(covariance, offset=1, axis1=-2, axis2=-1)
    sigma = np.sqrt(np.maximum(variance, 1e-30))
    expected = sigma*np.sqrt(2/np.pi)*np.exp(-delta**2/(2*sigma**2)) + delta*erf(delta/(np.sqrt(2)*sigma))
    return float(expected.mean())


def oracle_solver_check(config, steps, n_histories):
    gp, data = get_data(config)
    n = min(n_histories, len(data["test"]))
    mean, covariance = conditional_target(gp, data["test"])
    mean, covariance = mean[:n], covariance[:n]
    target = data["test"].target[:n]
    rng = np.random.default_rng(config["data_seed"] + 1793)
    normal = rng.standard_normal((n, gp.horizon))
    exact_samples = mean + np.einsum("nij,nj->ni", np.linalg.cholesky(covariance), normal)
    target_r = roughness(target)
    expected_r = gaussian_expected_roughness(mean, covariance)
    out = {"information_set": "privileged raw history and normalization statistics",
           "device": "CPU float64 analytic Gaussian matrices", "n_histories": n,
           "target_roughness": target_r, "exact_conditional_sample_roughness": roughness(exact_samples),
           "exact_conditional_expected_roughness": expected_r,
           "exact_samples_note": "True target-distribution samples; Cholesky coupling is not the ODE endpoint coupling. No paired endpoint error is computed against them.",
           "reference_note": "Conditional covariance comparison is analytic for every history, not estimated from a single generated path.",
           "sources": {}}
    for source in ("white", "gp"):
        source_cov = source_covariance(gp, source)
        _, _, _, eps = fixed_pairs(config, gp, data["test"], source, "test")
        paths, records = {}, []
        for k in steps:
            transfer, endpoint_cov = oracle_euler_transport(covariance, source_cov, k)
            paths[k] = mean + np.einsum("nij,nj->ni", transfer, eps[:n])
            r = roughness(paths[k])
            row = {"euler_steps": k, "generated_roughness": r,
                   "roughness_to_target": r / target_r,
                   "roughness_to_exact_conditional_sample": r / roughness(exact_samples),
                   "analytic_expected_roughness": gaussian_expected_roughness(mean, endpoint_cov),
                   "analytic_expected_roughness_to_exact": gaussian_expected_roughness(mean, endpoint_cov) / expected_r,
                   **covariance_diagnostics(endpoint_cov, covariance)}
            records.append(row)
        for row in records:
            row["paired_endpoint_rmse_to_largest_k"] = float(np.sqrt(np.mean((paths[row["euler_steps"]]-paths[max(steps)])**2)))
        out["sources"][source] = records
    out["interpretation"] = (
        "Near-singular smooth conditional covariance can make the exact regression field stiff near t=1. "
        "Euler can substantially suppress conditional variance while mean-path roughness appears accurate. "
        "Use these covariance/step-sensitivity diagnostics to distinguish solver effects from head quality; "
        "the original fixed-step primary results remain unchanged.")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-paths", type=int, default=256)
    parser.add_argument("--oracle-histories", type=int, default=128)
    parser.add_argument("--steps", default="32,64,128,256")
    parser.add_argument("--source", choices=("white", "gp"))
    parser.add_argument("--kind", choices=("mlp", "s4", "whitened_mlp"))
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()
    steps = sorted(set(int(k) for k in args.steps.split(",")))
    if not steps or min(steps) < 1 or args.n_paths < 1 or args.oracle_histories < 1:
        parser.error("Step counts, path count, and history count must be positive")
    device_name = args.device
    if device_name == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable to this process; do not silently fall back")
    torch.set_num_threads(4)
    device = torch.device(device_name)
    config = load_config(args.config)
    out = Path(args.output) if args.output else Path(args.results) / "robustness" / f"solver_seed{args.seed}.json"
    result = {"config_sha256": config_hash(config), "steps": steps,
              "status": "running", "primary_euler_steps_unchanged": config["euler_steps"],
              "protocol": "Fixed test histories and sources; no training, model selection, or primary-metric replacement.",
              "runs": {}}
    if not args.oracle_only:
        for source, kind in CELLS:
            if args.source and source != args.source:
                continue
            if args.kind and kind != args.kind:
                continue
            name, record = learned_solver_check(config, args.results, source, kind, args.seed,
                                                steps, args.n_paths, device)
            result["runs"][name] = record
            save_json(out, result)
    if not args.skip_oracle:
        result["oracle"] = oracle_solver_check(config, steps, args.oracle_histories)
    result["status"] = "complete"
    save_json(out, result)
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()
