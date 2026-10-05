"""Reproducible CLI: geometry -> train -> evaluation, with atomic run records."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import torch
from .data import (GPConfig, sample_dataset, sample_source, source_covariance,
                   interpolate, conditional_target, conditional_oracle_velocity,
                   conditional_oracle_noise)
from .metrics import fit_pca, spectrum, evaluate, velocity_metrics, oracle_noise_metrics

CELLS = [("white", "mlp"), ("white", "s4"), ("gp", "mlp"), ("gp", "s4"), ("gp", "whitened_mlp")]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def load_config(path):
    return json.loads(Path(path).read_text())


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def get_data(cfg):
    gp = GPConfig(**cfg["data"])
    seed = cfg["data_seed"]
    return gp, {split: sample_dataset(gp, cfg["n_" + split], seed + i)
                for i, split in enumerate(("train", "val", "test"))}


def geometry(cfg, output):
    """No model outcomes used to select the geometry. Test is reported only."""
    gp, data = get_data(cfg)
    out = Path(output)
    target_pca = fit_pca(data["train"].target)
    pcas, result = {}, {"config": asdict(gp), "data_seed": cfg["data_seed"],
                       "normalization": "official Sundial history-only population std; <=0.01 -> 1",
                       "matched_kernel_caveat": "same raw RBF kernel, not equal post-normalization marginal covariance",
                       "spectra": {"target": {}}, "pca": {"target": target_pca.to_dict()}}
    for split in ("train", "val", "test"):
        result["spectra"]["target"][split] = spectrum(data[split].target, target_pca)
    for j, source in enumerate(("white", "gp")):
        eps = sample_source(gp, cfg["n_train"], source, cfg["data_seed"] + 100 + j)
        pca = fit_pca(data["train"].target - eps)
        pcas[source] = pca
        result["pca"][source] = pca.to_dict()
        result["spectra"][source] = {}
        for i, split in enumerate(("train", "val", "test")):
            e = sample_source(gp, len(data[split]), source, cfg["data_seed"] + 100 + j + 10*i)
            result["spectra"][source][split] = spectrum(data[split].target - e, pca)
    g = result["spectra"]["gp"]["val"]
    target = result["spectra"]["target"]["val"]
    result["gate_pass"] = bool(g["top2_energy"] > .9 and target["mean_within_patch_std"] > .02 and target["roughness"] > .002)
    result["white_contrast_below_90pct"] = result["spectra"]["white"]["val"]["top2_energy"] < .9
    result["config_sha256"] = config_hash(cfg)
    save_json(out / "geometry.json", result)
    save_json(out / "config.json", cfg)
    return gp, data, pcas, result


def tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class CoordinateModel(torch.nn.Module):
    """Invertible linear whitening; no affine centering of velocity derivatives."""
    def __init__(self, base, pca=None):
        super().__init__()
        self.base = base
        f = base.horizon if hasattr(base, "horizon") else (len(pca.mean) if pca else 16)
        self.register_buffer("q", torch.eye(f) if pca is None else tensor(pca.components.T, "cpu"))
        self.register_buffer("scale", torch.ones(f) if pca is None else tensor(pca.scales, "cpu"))
    def to_coordinates(self, x):
        return (x @ self.q) / self.scale
    def from_coordinates(self, z):
        return (z * self.scale) @ self.q.T
    def forward_coordinates(self, x, t, h):
        return self.base(self.to_coordinates(x), t, h)
    def forward(self, x, t, h):
        return self.from_coordinates(self.forward_coordinates(x, t, h))


@torch.no_grad()
def predict(model, x, t, h, batch_size=1024):
    return torch.cat([model(x[i:i+batch_size], t[i:i+batch_size], h[i:i+batch_size])
                      for i in range(0, len(x), batch_size)])


@torch.no_grad()
def generate(model, eps, h, steps, batch_size=256):
    generated = []
    for start in range(0, len(eps), batch_size):
        x, history = eps[start:start+batch_size].clone(), h[start:start+batch_size]
        for k in range(steps):
            t = torch.full((len(x),), k / steps, device=x.device)
            x = x + model(x, t, history) / steps
        generated.append(x.cpu().numpy())
    return np.concatenate(generated)


def fixed_pairs(cfg, gp, ds, source, split):
    offset = {"val": 800, "test": 900}[split]
    # Same times, standard-normal source draws, and windows for all heads/seeds.
    eps = sample_source(gp, len(ds), source, cfg["data_seed"] + offset)
    times = np.random.default_rng(cfg["data_seed"] + offset + 1).uniform(0, 1, len(ds))
    x, u = interpolate(ds.target, eps, times)
    return x, u, times, eps


def train_cell(cfg, gp, data, pcas, source, kind, seed, output, device):
    from .models import create_model
    name = f"{source}_{kind}_seed{seed}"
    out = Path(output)
    result_path = out / "runs" / (name + ".json")
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old.get("config_sha256") != config_hash(cfg):
            raise RuntimeError(f"Refusing to reuse mismatched run: {result_path}")
        print("SKIP", name, flush=True)
        return old
    started = time.time()
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    base_kind = "mlp" if kind == "whitened_mlp" else kind
    base = create_model(base_kind, horizon=gp.horizon, lookback=gp.lookback, **cfg["model"])
    # Explicitly identical initial encoder state across heads at a given seed.
    torch.manual_seed(10000 + seed)
    for module in base.encoder.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    model = CoordinateModel(base, pcas[source] if kind == "whitened_mlp" else None).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Honor official S4 optimizer hints: no weight decay for SSM parameters.
    from .models import optimizer_groups
    optimizer = torch.optim.AdamW(optimizer_groups(model, cfg["learning_rate"], cfg["weight_decay"]))
    target, history = tensor(data["train"].target, device), tensor(data["train"].context, device)
    chol = tensor(np.linalg.cholesky(source_covariance(gp, source)), device)
    vx, vu, vt, _ = fixed_pairs(cfg, gp, data["val"], source, "val")
    vx, vu, vt, vh = [tensor(x, device) for x in (vx, vu, vt, data["val"].context)]
    # Training randomness reset after initialization; no architecture-dependent RNG consumption.
    torch.manual_seed(20000 + seed)
    if device.type == "mps":
        torch.mps.manual_seed(20000 + seed)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoints / (name + ".pt")
    best, best_step, curve = float("inf"), 0, []
    print(f"START {name} params={params} device={device} steps={cfg['steps']}", flush=True)
    for step in range(1, cfg["steps"] + 1):
        model.train()
        ix = torch.randint(len(target), (cfg["batch_size"],), device=device)
        y, h = target[ix], history[ix]
        eps = torch.randn(len(ix), gp.horizon, device=device) @ chol.T
        t = torch.rand(len(ix), device=device)
        x = (1-t[:, None])*eps + t[:, None]*y
        u = y-eps
        optimizer.zero_grad(set_to_none=True)
        pred = model.forward_coordinates(x, t, h)
        loss = (pred - model.to_coordinates(u)).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])
        optimizer.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            model.eval()
            with torch.no_grad():
                vp = predict(model, vx, vt, vh)
                vloss = (model.to_coordinates(vp-vu)).square().mean().item()
                raw = (vp-vu).square().mean().item()
            item = {"step": step, "train_loss": loss.item(), "validation_objective": vloss,
                    "validation_raw_mse": raw, "seconds": time.time()-started}
            if not np.isfinite(vloss):
                raise FloatingPointError(f"Non-finite validation loss: {name}")
            curve.append(item)
            if vloss < best:
                best, best_step = vloss, step
                torch.save({"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                            "config": cfg, "kind": kind, "source": source, "seed": seed,
                            "step": step, "pca": pcas[source].to_dict()}, ckpt)
            save_json(out / "training" / (name + ".json"), curve)
            print(f"{name} step={step} val={vloss:.5g} raw={raw:.5g} elapsed={item['seconds']:.1f}s", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    model.eval()
    # First use of final test labels for this selected checkpoint.
    tx, tu, tt, teps = fixed_pairs(cfg, gp, data["test"], source, "test")
    prediction = predict(model, tensor(tx, device), tensor(tt, device), tensor(data["test"].context, device)).cpu().numpy()
    n = min(cfg["n_paths"], len(tx))
    gen = generate(model, tensor(teps[:n], device), tensor(data["test"].context[:n], device), cfg["euler_steps"])
    metrics = evaluate(prediction, tu, gen, data["test"].target[:n], pcas[source])
    result = {"run": name, "source": source, "head": kind, "seed": seed, "parameters": params,
              "device": str(device), "torch": torch.__version__, "seconds": time.time()-started,
              "best_step": best_step, "best_validation_objective": best,
              "config_sha256": config_hash(cfg), "metrics": metrics, "curve": curve}
    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(samples / (name + ".npz"), generated=gen, target=data["test"].target[:n],
                        history=data["test"].context[:n], prediction=prediction, velocity=tu,
                        time=tt, source=teps[:n])
    save_json(result_path, result)
    print("DONE", name, "pc1", metrics["pc1_mse"], "res", metrics["residual_mse"],
          "rough_ratio", metrics["success"]["roughness_ratio"], flush=True)
    return result


def oracle_diagnostics(cfg, gp, data, pcas, output):
    from .data import sample_conditional_target
    out = Path(output)
    mu, cov = conditional_target(gp, data["test"])
    result = {"information_set": "privileged raw history; network receives normalized history only", "sources": {}}
    for source in ("white", "gp"):
        x,u,t,eps = fixed_pairs(cfg,gp,data["test"],source,"test")
        sc = source_covariance(gp,source)
        oracle = conditional_oracle_velocity(x,t,mu,cov,sc)
        noise = conditional_oracle_noise(t,cov,sc)
        n = min(cfg["n_paths"],len(x))
        # Exact conditional target samples, not approximate learned/ODE samples.
        gen = sample_conditional_target(gp,data["test"],seed=cfg["data_seed"]+1100)[:n, 0]
        record = evaluate(oracle,u,gen,data["test"].target[:n],pcas[source])
        record["irreducible_noise"] = oracle_noise_metrics(noise,pcas[source])
        record["zero_velocity"] = velocity_metrics(np.zeros_like(u),u,pcas[source])
        record["mean_velocity"] = velocity_metrics(np.broadcast_to(pcas[source].mean,u.shape),u,pcas[source])
        result["sources"][source] = record
    save_json(out/"oracle.json",result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=["geometry","train","oracle"])
    parser.add_argument("--config",default="configs/main.json")
    parser.add_argument("--output",default="results/main")
    parser.add_argument("--device",default="mps",choices=["mps","cpu"])
    parser.add_argument("--cell",default="all")
    parser.add_argument("--seed",type=int)
    args=parser.parse_args()
    cfg=load_config(args.config)
    out=Path(args.output)
    # Refuse accidental mixing of distinct experiments in one result directory.
    if (out/"config.json").exists() and config_hash(load_config(out/"config.json")) != config_hash(cfg):
        raise ValueError("Output directory contains a different configuration")
    gp,data,pcas,geo=geometry(cfg,out)
    if args.command=="geometry":
        print(json.dumps({"gate":geo["gate_pass"],"top2_validation":{k:v["val"]["top2_energy"] for k,v in geo["spectra"].items()}},indent=2))
        return
    if not geo["gate_pass"]:
        raise RuntimeError("Task1 geometry gate failed; training is forbidden")
    if args.command=="oracle":
        oracle_diagnostics(cfg,gp,data,pcas,out)
        return
    device=torch.device(args.device)
    if args.device=="mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable to this process. Do not silently fall back to CPU; on macOS sandbox requires external GPU permission.")
    torch.set_num_threads(4)
    save_json(out/"environment.json",{"platform":platform.platform(),"torch":torch.__version__,
                                      "numpy":np.__version__,"device":str(device),"mps_available":torch.backends.mps.is_available()})
    cells=CELLS if args.cell=="all" else [tuple(args.cell.split(":",1))]
    seeds=cfg["seeds"] if args.seed is None else [args.seed]
    for seed in seeds:
        for source,kind in cells:
            train_cell(cfg,gp,data,pcas,source,kind,seed,out,device)


if __name__ == "__main__":
    main()
