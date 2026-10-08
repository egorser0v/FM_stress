"""Actual CPU/MPS parity for the five original-setting mixture-study heads.

Five CPU optimizer steps make each compared checkpoint nonzero; MPS receives
exactly that state. This is numerical QA, not another scored experiment.
"""
from pathlib import Path
import copy
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fm_stress.data import GPConfig, sample_dataset
from fm_stress.experiment import load_config, save_json
from fm_stress.metrics import PCA
from fm_stress.prior_mixture import make_model, objective, source_draw, source_name
from fm_stress.prior_mixture_diagnostics import gradient_diagnostics


def compare(a, b, atol=2e-5, rtol=5e-4):
    a = torch.as_tensor(a).detach().cpu().double()
    b = torch.as_tensor(b).detach().cpu().double()
    maximum = float((a - b).abs().max())
    tolerance = atol + rtol * float(a.abs().max())
    return {"max_abs_difference": maximum, "tolerance": tolerance,
            "passed": bool(torch.isfinite(a).all() and torch.isfinite(b).all()
                           and maximum <= tolerance)}


def checked_diagnostics(model, x, t, history, velocity, pca):
    model.train()
    model.base.encoder.eval()  # Deliberately mixed flags must survive.
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 0.125)
    states = [module.training for module in model.modules()]
    gradients = [(parameter.grad, parameter.grad.clone()) for parameter in model.parameters()]
    cpu_rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    value = gradient_diagnostics(model, x, t, history, velocity, pca)
    checks = {
        "module_flags_preserved": states == [module.training for module in model.modules()],
        "grad_objects_and_values_preserved": all(
            parameter.grad is original and torch.equal(parameter.grad, snapshot)
            for parameter, (original, snapshot) in zip(model.parameters(), gradients)),
        "cpu_rng_preserved": torch.equal(cpu_rng, torch.get_rng_state()),
        "mps_rng_preserved": torch.equal(mps_rng, torch.mps.get_rng_state()),
    }
    return value, checks


def loss_and_grads(model, x, t, h, u):
    model.eval()
    model.zero_grad(set_to_none=True)
    prediction = model(x, t, h)
    loss = objective(model, prediction, u)
    loss.backward()
    grads = {name: parameter.grad.detach().cpu().clone() if parameter.grad is not None else None
             for name, parameter in model.named_parameters()}
    return prediction.detach(), loss.detach(), grads


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("Actual MPS required; no CPU fallback")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable MPS operator fallback for this audit")
    torch.set_num_threads(2)
    torch.empty(1, device="mps")
    cfg = load_config(ROOT / "configs/prior_mixture.json")
    geometry = load_config(ROOT / f"results/prior-mixture/dataset_{cfg['data_seeds'][0]}/geometry.json")
    ds = sample_dataset(GPConfig(**cfg["data"]), 16, 910731)
    history = torch.tensor(ds.context, dtype=torch.float32)
    target = torch.tensor(ds.target, dtype=torch.float32)
    times = torch.linspace(0.05, 0.95, len(history))
    rows = []
    for alpha in (0.0, 0.5, 1.0):
        p = geometry["sources"][source_name(alpha)]["pca"]
        pca = PCA(np.array(p["mean"]), np.array(p["components"]),
                  np.array(p["eigenvalues"]), p["floor"])
        source = torch.tensor(source_draw(cfg, len(history), alpha, 910732), dtype=torch.float32)
        x = (1 - times[:, None]) * source + times[:, None] * target
        velocity = target - source
        for head in ("mlp", "s4", "whitened_mlp", "balanced_mlp", "whitened_raw_mlp"):
            cpu = make_model(cfg, head, pca, 941, torch.device("cpu"))
            opt = torch.optim.AdamW(cpu.parameters(), lr=0.001, weight_decay=0.0001)
            for _ in range(5):
                opt.zero_grad(set_to_none=True)
                loss = objective(cpu, cpu(x, times, history), velocity)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(cpu.parameters(), 1.0)
                opt.step()
            mps = copy.deepcopy(cpu).to("mps")
            xm, tm, hm, um = [v.to("mps") for v in (x, times, history, velocity)]
            a, loss_a, ga = loss_and_grads(cpu, x, times, history, velocity)
            b, loss_b, gb = loss_and_grads(mps, xm, tm, hm, um)
            if a.abs().max().item() < 1e-5:
                raise AssertionError("Trivial zero output cannot establish parity")
            gradients = {}
            for name in ga:
                if ga[name] is None or gb[name] is None:
                    gradients[name] = {"passed": ga[name] is None and gb[name] is None,
                                       "unused": True}
                else:
                    gradients[name] = compare(ga[name], gb[name], atol=2e-4, rtol=1e-3)
            dc, ic = checked_diagnostics(cpu, x, times, history, velocity, pca)
            dm, im = checked_diagnostics(mps, xm, tm, hm, um, pca)
            diagnostics = {"raw_total_gradient_norm": compare(
                dc["raw_total_gradient_norm"], dm["raw_total_gradient_norm"])}
            for group in ("pc1", "pc2", "residual"):
                for key in ("gradient_norm", "raw_mse_contribution"):
                    diagnostics[f"{group}.{key}"] = compare(dc["groups"][group][key],
                                                            dm["groups"][group][key])
            for pair in dc["gradient_cosines"]:
                va, vb = dc["gradient_cosines"][pair], dm["gradient_cosines"][pair]
                diagnostics[pair] = ({"passed": va is None and vb is None} if va is None or vb is None
                                     else compare(va, vb, atol=2e-3, rtol=1e-3))
            roundtrips = {"cpu": compare(x, cpu.inverse(cpu.coordinates(x)), atol=2e-5, rtol=1e-5),
                          "mps": compare(xm, mps.inverse(mps.coordinates(xm)), atol=2e-5, rtol=1e-5)}
            checks = {"output": compare(a, b), "optimized_objective": compare(loss_a, loss_b),
                      **roundtrips}
            for label, diag in (("cpu", dc), ("mps", dm)):
                checks[f"{label}_gradient_additivity"] = {
                    "passed": diag["gradient_reconstruction_absolute_error"] <=
                    1e-5 + 2e-5 * diag["raw_total_gradient_norm"],
                    "absolute_error": diag["gradient_reconstruction_absolute_error"]}
            row = {"head": head, "alpha": alpha, "trained_steps_cpu": 5,
                   "cpu_output_max_abs": float(a.abs().max()), "checks": checks,
                   "parameter_gradients": gradients, "raw_gradient_diagnostics": diagnostics,
                   "cpu_invariance": ic, "mps_invariance": im,
                   "cpu_diagnostics": dc, "mps_diagnostics": dm}
            row["passed"] = (all(v["passed"] for group in (checks, gradients, diagnostics)
                                 for v in group.values()) and all(ic.values()) and all(im.values()))
            rows.append(row)
            print(alpha, head, "PASS" if row["passed"] else "FAIL", flush=True)
    files = ("fm_stress/prior_mixture.py", "fm_stress/prior_mixture_diagnostics.py",
             "fm_stress/models.py", "scripts/check_prior_mixture_device.py")
    result = {"status": "passed" if all(r["passed"] for r in rows) else "failed",
              "device": "mps", "fallback_enabled": False, "torch": torch.__version__,
              "rows": rows, "comparison_count": len(rows),
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
              "scope": "Five heads across white, half-GP covariance mixture, and GP; five CPU steps followed by identical CPU/MPS checkpoints. Forward, actual raw/balanced optimization loss and gradients, raw-MSE diagnostic norms/cosines, transform roundtrip, state/.grad/RNG preservation. No scientific training result.",
              "interpretation": "Diagnostic gradients always use raw MSE, including heads trained/selected with balanced objective. Cross-parameterization norms are not invariant; selected checkpoint is chosen by each head's own objective."}
    save_json(ROOT / "results/diagnostics/prior_mixture_device.json", result)
    if result["status"] != "passed":
        raise AssertionError("Parity check failed; inspect diagnostic JSON")


if __name__ == "__main__":
    main()
