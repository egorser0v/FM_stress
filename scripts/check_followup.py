"""Independent CPU-only completion audit of both supplemental experiments.

Read-only with respect to experiment artifacts; writes one verification record
only after every required check succeeds. Run from the project root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from fm_stress.data import GPConfig, sample_dataset, sample_source, sample_conditional_target
from fm_stress.experiment import config_hash, fixed_pairs
from fm_stress.metrics import fit_pca, evaluate
from fm_stress.controlled_followup import HEADS, LOSSES, hashes, make_model, pilot_name
from fm_stress.ensemble_evaluation import context_scores, shared_inputs


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, rtol=1e-11, atol=1e-12):
    if isinstance(expected, dict):
        require(set(actual) == set(expected), f"{label}: different keys")
        for key in expected:
            close(actual[key], expected[key], f"{label}.{key}", rtol, atol)
    elif expected is None or isinstance(expected, (str, bool)):
        require(actual == expected, f"{label}: {actual!r} != {expected!r}")
    else:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=label)


def bootstrap(differences, seed, repetitions):
    """Independent expression of the declared paired percentile bootstrap."""
    values = np.asarray(differences, dtype=float)
    ix = np.random.default_rng(seed).integers(0, len(values), size=(repetitions, len(values)))
    draws = np.mean(values[ix], axis=1)
    return {"mean_difference": float(values.mean()), "ci_low": float(np.quantile(draws, .025)),
            "ci_high": float(np.quantile(draws, .975)), "pairs": len(values)}


def verify_curve(record, expected_steps):
    curve = record["curve"]
    require(curve[-1]["step"] == expected_steps, f"{record['run']}: training not complete")
    chosen = min(curve, key=lambda x: x["validation_objective"])
    require(record["best_step"] == chosen["step"], f"{record['run']}: best step differs from first argmin")
    require(record["best_validation_objective"] == chosen["validation_objective"], f"{record['run']}: best objective mismatch")


def check_controlled(root, analysis):
    cfg, selection = read(root/"config.json"), read(root/"selection.json")
    require(read(root/"source_manifest.json") == hashes(), "Controlled actual source hashes changed")
    require(selection["config_sha256"] == config_hash(cfg), "Controlled selection config mismatch")
    expected_pilots = {pilot_name(h, loss, lr) for h in HEADS for loss in LOSSES for lr in cfg["pilot_learning_rates"]}
    require(set(selection["pilot_record_sha256"]) == expected_pilots, "Selection does not bind exactly eight pilots")
    require({p.stem for p in (root/"pilots"/"runs").glob("*.json")} == expected_pilots, "Unexpected pilot run set")
    for h in HEADS:
        for loss in LOSSES:
            candidates = []
            for lr in cfg["pilot_learning_rates"]:
                name = pilot_name(h, loss, lr)
                path = root/"pilots"/"runs"/(name+".json")
                require(sha(path) == selection["pilot_record_sha256"][name], "Pilot record hash mismatch")
                r = read(path)
                require(r["head"] == h and r["loss"] == loss and r["learning_rate"] == lr, "Pilot identity mismatch")
                require(r["seed"] == cfg["pilot_seed"] and r["config_sha256"] == config_hash(cfg), "Pilot seed/config mismatch")
                require(r["phase"] == "pilot" and r["status"] == "complete" and r["device"] == "mps", "Invalid pilot status")
                require("metrics" not in r and not (root/"pilots"/"samples"/(name+".npz")).exists(), "Pilot test evaluation detected")
                verify_curve(r, cfg["pilot_steps"])
                checkpoint = torch.load(root/"pilots"/"checkpoints"/(name+".pt"), map_location="cpu", weights_only=False)
                require(checkpoint["step"] == r["best_step"] and checkpoint["config"] == cfg, "Pilot checkpoint mismatch")
                candidates.append(r)
            best = min(candidates, key=lambda x: (x["best_validation_objective"], x["learning_rate"]))
            require(selection["selected"][h+"_"+loss] == {
                "learning_rate": best["learning_rate"], "pilot_validation_objective": best["best_validation_objective"],
                "pilot_run": best["run"]}, "Locked LR is not validation argmin with stated tie rule")
    expected_runs = {f"gp_{h}_{loss}_seed{s}" for h in HEADS for loss in LOSSES for s in cfg["seeds"]}
    require({p.stem for p in (root/"runs").glob("*.json")} == expected_runs, "Controlled final suite incomplete or contains extras")
    gp = GPConfig(**cfg["data"])
    train = sample_dataset(gp, cfg["n_train"], cfg["data_seed"])
    eps = sample_source(gp, cfg["n_train"], "gp", cfg["data_seed"]+101)
    pca = fit_pca(train.target-eps)
    target_pca = fit_pca(train.target)
    geometry = read(root/"geometry.json")
    require(geometry["gate_pass"], "Controlled geometry gate failed")
    close(geometry["pca"]["gp"], pca.to_dict(), "Regenerated train PCA")
    test = sample_dataset(gp, cfg["n_test"], cfg["data_seed"]+2)
    tx, velocity, times, sources = fixed_pairs(cfg, gp, test, "gp", "test")
    rows = []
    path_moments = []
    max_prediction_difference = 0.
    for name in sorted(expected_runs):
        record = read(root/"runs"/(name+".json"))
        h, loss, seed = record["head"], record["loss"], record["seed"]
        require(record["run"] == f"gp_{h}_{loss}_seed{seed}" and record["config_sha256"] == config_hash(cfg), "Final identity mismatch")
        require(record["device"] == "mps" and record["status"] == "complete" and record["phase"] == "final", "Final status mismatch")
        require(record["learning_rate"] == selection["selected"][h+"_"+loss]["learning_rate"], "Final LR mismatch")
        verify_curve(record, cfg["steps"])
        checkpoint = torch.load(root/"checkpoints"/(name+".pt"), map_location="cpu", weights_only=False)
        require(checkpoint["step"] == record["best_step"] and checkpoint["config"] == cfg, "Final checkpoint step/config mismatch")
        require(checkpoint["encoder"] == "identity" and not any(k.startswith("encoder.") for k in checkpoint["state_dict"]), "Encoder is not fixed identity")
        require((checkpoint["head"], checkpoint["loss"], checkpoint["seed"]) == (h, loss, seed), "Checkpoint identity mismatch")
        with np.load(root/"samples"/(name+".npz")) as a:
            n = len(a["generated"])
            require(n == min(cfg["n_paths"], cfg["n_test"]), "Wrong path count")
            for key, expected in {"target": test.target[:n], "history": test.context[:n], "source": sources[:n],
                                  "velocity": velocity, "time": times}.items():
                close(a[key], expected, f"{name}: regenerated {key}", rtol=0, atol=0)
            metrics = evaluate(a["prediction"], a["velocity"], a["generated"], a["target"], pca)
            close(record["metrics"], metrics, f"{name}: all archived metrics")
            generated_pc = (a["generated"]-target_pca.mean) @ target_pca.components.T
            observed_pc = (a["target"]-target_pca.mean) @ target_pca.components.T
            generated_moment = np.mean(generated_pc**2,axis=0)
            target_moment = np.mean(observed_pc**2,axis=0)
            tail_ratio = float(generated_moment[6:].sum()/target_moment[6:].sum())
            path_moments.append({"run":name,"head":h,"loss":loss,"seed":seed,
                "generated_pc_second_moments":generated_moment.tolist(),
                "target_pc_second_moments":target_moment.tolist(),
                "generated_pc7_16_second_moment":float(generated_moment[6:].sum()),
                "target_pc7_16_second_moment":float(target_moment[6:].sum()),
                "pc7_16_second_moment_ratio":tail_ratio})
            model = make_model(cfg, h, seed, torch.device("cpu"))
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            with torch.no_grad():
                prediction = model(torch.tensor(tx[:64], dtype=torch.float32), torch.tensor(times[:64], dtype=torch.float32),
                                   torch.tensor(test.context[:64], dtype=torch.float32)).numpy()
            difference = float(np.max(np.abs(prediction-a["prediction"][:64])))
            max_prediction_difference = max(max_prediction_difference, difference)
            np.testing.assert_allclose(prediction, a["prediction"][:64], atol=5e-5, rtol=5e-5,
                                       err_msg=f"{name}: checkpoint does not reproduce saved predictions")
        rows.append({"head": h, "loss": loss, "seed": seed, **metrics,
                     "roughness_ratio": metrics["success"]["roughness_ratio"],
                     "target_pca_pc7_16_moment_ratio":tail_ratio})
    summary = read(analysis/"summary.json")
    require(summary["verified_final_runs"] == 12 and summary["verified_pilots"] == 8, "Wrong controlled analysis counts")
    close(summary["target_pca"],target_pca.to_dict(),"Analysis train-target PCA")
    require(len(summary["target_pca_path_moments"]) == 12,"Missing target-PCA path rows")
    for expected in path_moments:
        actual = next(r for r in summary["target_pca_path_moments"] if r["run"] == expected["run"])
        close(actual,expected,"Generated target-PC moments")
    for cell in summary["aggregate"]:
        group = [r for r in rows if r["head"] == cell["head"] and r["loss"] == cell["loss"]]
        require(cell["n"] == len(group), "Aggregate count mismatch")
        for metric in cell.keys()-{"head", "loss", "n"}:
            values = [r[metric] for r in group]
            close(cell[metric], {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}, "Controlled means/SD")
    def values(h, loss, metric):
        return np.array([next(r[metric] for r in rows if r["head"]==h and r["loss"]==loss and r["seed"]==s) for s in cfg["seeds"]])
    for contrast in summary["contrasts"]:
        label, metric = contrast["contrast"], contrast["metric"]
        if label.startswith("S4 minus MLP"):
            loss = label.split("(")[1].rstrip(")")
            diff = values("s4",loss,metric)-values("mlp",loss,metric)
        elif label.startswith("balanced minus raw"):
            head = label.split("(")[1].rstrip(")")
            diff = values(head,"balanced",metric)-values(head,"raw",metric)
        else:
            require(label == "head-by-loss interaction: balanced head gap minus raw head gap", "Unknown contrast")
            diff = values("s4","balanced",metric)-values("mlp","balanced",metric)-values("s4","raw",metric)+values("mlp","raw",metric)
        close(contrast["paired_values"], diff, "Paired values")
        for key,value in bootstrap(diff,902,10000).items():
            close(contrast[key], value, "Controlled paired bootstrap "+key)
    return {"pilots": 8, "final_runs": 12, "locked_lr_argmin_verified": True,
            "all_metrics_recomputed": True, "fresh_split_and_source_exact_match": True,
            "checkpoint_cpu_prediction_max_difference": max_prediction_difference,
            "aggregate_and_bootstrap_recomputed": True, "actual_source_hashes_verified": True}


def check_ensemble(root):
    cfg, manifest, summary = read(root/"config.json"), read(root/"manifest.json"), read(root/"summary.json")
    require(summary["complete"], "Ensemble suite incomplete")
    require(manifest["config"] == cfg, "Ensemble manifest/config disagreement")
    unsigned = {k:v for k,v in manifest.items() if k != "fingerprint"}
    require(config_hash(unsigned) == manifest["fingerprint"] == summary["fingerprint"], "Invalid ensemble fingerprint")
    training = read(cfg["training_config"])
    require(config_hash(training) == manifest["training_config_sha256"], "Training config hash mismatch")
    for filename,digest in manifest["source_sha256"].items():
        require(sha(Path("fm_stress")/filename) == digest, "Actual ensemble numerical source changed")
    for name,digest in manifest["checkpoint_sha256"].items():
        require(sha(Path(cfg["checkpoint_directory"])/(name+".pt")) == digest, "Actual checkpoint hash changed")
    gp = GPConfig(**training["data"])
    ds,sources = shared_inputs(gp,cfg)
    with np.load(root/"common_inputs.npz") as a:
        for key,expected in {"history":ds.context,"target":ds.target,"raw_history":ds.raw_context,"source":sources}.items():
            close(a[key], expected, "Fresh ensemble "+key, rtol=0, atol=0)
    names = {f"gp_{h}_seed{s}" for h in cfg["heads"] for s in cfg["seeds"]}
    require({p.stem for p in (root/"runs").glob("*.json")} == names and len(names)==15, "Ensemble run set mismatch")
    records=[]
    for name in sorted(names):
        r=read(root/"runs"/(name+".json"))
        require(r["fingerprint"] == manifest["fingerprint"] and r["device"] == "mps", "Ensemble provenance mismatch")
        with np.load(root/"samples"/(name+".npz")) as a:
            require(a["generated"].shape == sources.shape, "Ensemble sample shape mismatch")
            metrics=context_scores(a["generated"],ds.target)
        close(r["per_context"],{k:v.tolist() for k,v in metrics.items()},name+" per-context scores")
        close(r["means"],{k:float(v.mean()) for k,v in metrics.items()},name+" means")
        records.append(r)
    groups={h:[r for r in records if r["head"]==h] for h in cfg["heads"]}
    for h,group in groups.items():
        expected={"n_seeds":len(group),"metrics":{m:{"mean":float(np.mean([r["means"][m] for r in group])),
                  "seed_sd":float(np.std([r["means"][m] for r in group],ddof=1))} for m in group[0]["means"]}}
        close(summary["heads"][h],expected,"Ensemble summary means/SD")
    require(len(summary["contrasts"]) == 2*len(records[0]["per_context"]), "Ensemble contrast count mismatch")
    for contrast in summary["contrasts"]:
        av=np.mean([r["per_context"][contrast["metric"]] for r in groups[contrast["a"]]],axis=0)
        bv=np.mean([r["per_context"][contrast["metric"]] for r in groups[contrast["b"]]],axis=0)
        for key,value in bootstrap(av-bv,cfg["bootstrap_seed"],cfg["bootstrap_repetitions"]).items():
            close(contrast[key],value,"Ensemble paired bootstrap "+key)
    if cfg["include_privileged_reference"]:
        reference=sample_conditional_target(gp,ds,samples=cfg["ensemble_size"],seed=cfg["reference_seed"])
        with np.load(root/"samples"/"privileged_reference.npz") as a:
            close(a["generated"],reference.astype(np.float32),"Privileged exact draws",rtol=0,atol=0)
        ref=read(root/"privileged_reference.json")
        require(ref["fingerprint"] == manifest["fingerprint"], "Reference fingerprint mismatch")
        metrics=context_scores(reference,ds.target)
        close(ref["per_context"],{k:v.tolist() for k,v in metrics.items()},"Privileged per-context metrics")
        close(ref["means"],{k:float(v.mean()) for k,v in metrics.items()},"Privileged means")
    solver_count = 0
    if cfg.get("solver_sensitivity"):
        options = cfg["solver_sensitivity"]
        n, seed, steps = options["n_histories"], options["seed"], options["euler_steps"]
        directory = root/"solver_sensitivity"
        solver_summary = read(directory/"summary.json")
        require(solver_summary["fingerprint"] == manifest["fingerprint"], "Solver summary fingerprint mismatch")
        require(len(solver_summary["rows"]) == len(cfg["heads"]), "Incomplete solver summary")
        for head in cfg["heads"]:
            name = f"gp_{head}_seed{seed}"
            r = read(directory/(name+".json"))
            require(r["fingerprint"] == manifest["fingerprint"] and r["device"] == "mps", "Solver provenance mismatch")
            require(r["n_histories"] == n and r["euler_steps"] == steps and r["baseline_euler_steps"] == cfg["euler_steps"], "Solver protocol mismatch")
            with np.load(directory/(name+".npz")) as a:
                new = a["generated"].copy()
            with np.load(root/"samples"/(name+".npz")) as a:
                old = a["generated"][:n].copy()
            require(new.shape == old.shape, "Solver sample shape mismatch")
            old_scores, new_scores = context_scores(old, ds.target[:n]), context_scores(new, ds.target[:n])
            close(r["baseline_means"], {k:float(v.mean()) for k,v in old_scores.items()}, "Solver baseline scores")
            close(r["new_means"], {k:float(v.mean()) for k,v in new_scores.items()}, "Solver new scores")
            close(r["paired_context_metric_differences"], {k:(new_scores[k]-old_scores[k]).tolist() for k in new_scores}, "Solver paired changes")
            close(r["paired_endpoint_rmse"], float(np.sqrt(np.square(new.astype(float)-old).mean())), "Solver endpoint RMSE")
            require(next(x for x in solver_summary["rows"] if x["name"] == name) == r, "Solver summary not equal to record")
            solver_count += 1
    return {"runs":15,"fresh_inputs_exact_match":True,"all_per_context_metrics_recomputed":True,
            "summary_and_paired_bootstrap_recomputed":True,"actual_source_and_checkpoint_hashes_verified":True,
            "privileged_reference_regenerated":bool(cfg["include_privileged_reference"]),
            "solver_sensitivity_runs_recomputed":solver_count}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/diagnostics/followup_verification.json")
    parser.add_argument("--suite",choices=("both","controlled","ensemble"),default="both")
    args=parser.parse_args()
    torch.set_num_threads(4)
    result={"execution_device":"cpu","status":"passed","scope":args.suite}
    if args.suite in ("both","controlled"):
        result["controlled"]=check_controlled(Path("results/controlled_followup"),Path("output/controlled-followup"))
    if args.suite in ("both","ensemble"):
        result["ensemble"]=check_ensemble(Path("results/ensemble-evaluation"))
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
