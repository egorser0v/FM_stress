#!/usr/bin/env python3
"""Independent NumPy reconstruction and CPU replay of the prior-mixture study.

No production data sampling, PCA fitting or metric routines are used in the
independent calculations below. Checkpoint replay necessarily reuses the model
architecture; it verifies saved predictions, not the stochastic training path.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def close(actual, expected, label, atol=1e-9, rtol=1e-8):
    try:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    except AssertionError as exc:
        raise AssertionError(label + ': ' + str(exc)) from exc


def independent_dataset(data_config, n, seed):
    """Rebuild the joint GP and normalize with history-only population statistics."""
    l, f = data_config['lookback'], data_config['horizon']
    grid = np.arange(l + f)
    covariance = data_config.get('variance', 1.) * np.exp(
        -(grid[:, None] - grid[None, :]) ** 2 / (2 * data_config['lengthscale'] ** 2))
    covariance += data_config.get('nugget', 1e-6) * np.eye(l + f)
    raw = np.random.default_rng(seed).standard_normal((n, l + f)) @ np.linalg.cholesky(covariance).T
    mean = raw[:, :l].mean(axis=1, keepdims=True)
    scale = raw[:, :l].std(axis=1, ddof=0, keepdims=True)
    scale = np.where(scale > data_config.get('normalization_threshold', .01), scale, 1.)
    normalized = (raw - mean) / scale
    return dict(context=normalized[:, :l], target=normalized[:, l:], raw=raw,
                location=mean, scale=scale)


def independent_covariance(data_config, alpha, unit_diagonal=True):
    f = data_config['horizon']
    grid = np.arange(f)
    gp = data_config.get('variance', 1.) * np.exp(
        -(grid[:, None] - grid[None, :]) ** 2 / (2 * data_config['lengthscale'] ** 2))
    gp += data_config.get('nugget', 1e-6) * np.eye(f)
    if unit_diagonal:
        gp = gp / (data_config.get('variance', 1.) + data_config.get('nugget', 1e-6))
    return (1. - alpha) * np.eye(f) + alpha * gp


def independent_pca_check(training_values, pca, label):
    q, values = np.asarray(pca['components']), np.asarray(pca['eigenvalues'])
    x = np.asarray(training_values, float)
    close(pca['mean'], x.mean(0), label + ' training mean')
    close(q @ q.T, np.eye(x.shape[1]), label + ' orthonormal axes', atol=2e-10)
    cov = np.cov(x, rowvar=False, ddof=1)
    close(q @ cov @ q.T, np.diag(values), label + ' diagonalization', atol=2e-8)
    if np.any(np.diff(values) > 1e-10):
        raise AssertionError(label + ' unordered eigenvalues')
    close(pca['floor'], max(values[0] * 1e-6, 1e-8), label + ' variance floor')


def independent_velocity(prediction, truth, pca):
    error = np.asarray(prediction, float) - np.asarray(truth, float)
    pc = np.square(error @ np.asarray(pca['components']).T).mean(axis=0)
    normalized = pc / np.maximum(pca['eigenvalues'], pca['floor'])
    return dict(velocity_mse=float(np.square(error).mean()), pc_mse=pc.tolist(),
                pc1_mse=float(pc[0]), pc2_mse=float(pc[1]), residual_mse=float(pc[2:].mean()),
                pc_normalized_mse=normalized.tolist(), pc1_normalized_mse=float(normalized[0]),
                pc2_normalized_mse=float(normalized[1]), residual_normalized_mse=float(normalized[2:].mean()),
                variance_floor=float(pca['floor']))


def independent_scores(generated, targets):
    """Explicit unordered-pair fair CRPS and unscaled Euclidean energy score."""
    x, y = np.asarray(generated, float), np.asarray(targets, float)
    if x.ndim != 3 or y.shape != (x.shape[0], x.shape[2]) or x.shape[1] < 2:
        raise AssertionError('Forecast arrays must be [contexts,draws,horizon] and [contexts,horizon]')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise AssertionError('Nonfinite forecast arrays')
    n, samples, horizon = x.shape
    first, second = np.triu_indices(samples, 1)
    pair_delta = x[:, first] - x[:, second]
    crps = np.abs(x-y[:, None]).mean((1, 2)) - np.abs(pair_delta).sum((1, 2))/(samples*(samples-1)*horizon)
    energy = np.linalg.norm(x-y[:, None], axis=2).mean(1) - np.linalg.norm(pair_delta, axis=2).sum(1)/(samples*(samples-1))
    result = dict(fair_crps=crps, fair_energy_score=energy,
                  mean_forecast_mse=np.square(x.mean(1)-y).mean(1),
                  ensemble_variance=x.var(1, ddof=1).mean(1),
                  generated_roughness=np.abs(np.diff(x, axis=2)).mean((1, 2)),
                  target_roughness=np.abs(np.diff(y, axis=1)).mean(1))
    for level in (.8, .9):
        lo, hi = np.quantile(x, [(1-level)/2, (1+level)/2], axis=1, method='linear')
        result[f'coverage_{int(level*100)}'] = ((y >= lo) & (y <= hi)).mean(1)
        result[f'width_{int(level*100)}'] = (hi-lo).mean(1)
    return result


def replay(model, x, times, history, batch_size=256):
    result = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            args = [torch.as_tensor(v[start:start+batch_size], dtype=torch.float32)
                    for v in (x, times, history)]
            result.append(model(*args).numpy())
    return np.concatenate(result)


def independent_euler(model, source, history, steps):
    with torch.no_grad():
        x = torch.as_tensor(source, dtype=torch.float32).clone()
        h = torch.as_tensor(history, dtype=torch.float32)
        for step in range(steps):
            time = torch.full((len(x),), step/steps, dtype=torch.float32)
            x += model(x, time, h)/steps
        return x.numpy()


def compare_mapping(values, record, label, atol=1e-9, rtol=1e-8):
    for key, value in values.items():
        close(record[key], value, label + ' ' + key, atol=atol, rtol=rtol)


def independent_spectrum(values, pca):
    x = np.asarray(values, float)
    variances = (x @ np.asarray(pca['components']).T).var(0, ddof=1)
    energy = variances/variances.sum()
    positive = energy[energy > 0]
    return dict(pc_variance=variances.tolist(), energy=energy.tolist(),
                top1_energy=float(energy[0]), top2_energy=float(energy[:2].sum()),
                total_variance=float(variances.sum()),
                effective_rank=float(np.exp(-(positive*np.log(positive)).sum())),
                roughness=float(np.abs(np.diff(x, axis=1)).mean()),
                mean_within_patch_std=float(x.std(1).mean()))


def independent_success(pc1, residual, generated_roughness, target_roughness):
    error_pass = bool(pc1 > 0 and .8*pc1 <= residual <= 1.2*pc1)
    roughness_pass = bool(target_roughness > 0 and .8*target_roughness <= generated_roughness <= 1.2*target_roughness)
    return dict(residual_to_pc1=None if pc1 == 0 else residual/pc1,
                roughness_ratio=None if target_roughness == 0 else generated_roughness/target_roughness,
                velocity_rule_pass=error_pass, roughness_rule_pass=roughness_pass,
                works=error_pass and roughness_pass)


def draw(covariance, n, seed):
    return np.random.default_rng(seed).standard_normal((n, len(covariance))) @ np.linalg.cholesky(covariance).T


def train_pca(values):
    """Independent eigendecomposition for source-spectrum validation only."""
    covariance = np.cov(values, rowvar=False)
    vals, vectors = np.linalg.eigh(covariance)
    order = np.argsort(vals)[::-1]
    return dict(components=vectors[:, order].T.tolist(), eigenvalues=np.maximum(vals[order], 0).tolist())


def validate_geometry(cfg, data_seed, geometry):
    datasets = {split: independent_dataset(cfg['data'], cfg['n_'+split], data_seed+i)
                for i, split in enumerate(('train', 'val', 'test'))}
    target_pca = geometry['target_pca']
    independent_pca_check(datasets['train']['target'], target_pca, f'{data_seed} target')
    compare_mapping(independent_spectrum(datasets['val']['target'], target_pca),
                    geometry['validation_target'], f'{data_seed} target spectrum')
    if geometry['test_inspected'] is not False:
        raise AssertionError('Test data marked inspected in pretraining geometry')
    for split in ('train', 'val'):
        for field, key in (('history', 'context'), ('target', 'target')):
            values = np.ascontiguousarray(datasets[split][key])
            if hashlib.sha256(values.tobytes()).hexdigest() != geometry['split_hashes'][split][field]:
                raise AssertionError(f'{data_seed} independent {split} {field} hash differs')
    expected_names = {'white' if a == 0 else 'gp' if a == 1 else f'mix{round(a*100):03d}' for a in cfg['alphas']}
    if set(geometry['sources']) != expected_names:
        raise AssertionError('Source geometry coverage differs')
    for name, row in geometry['sources'].items():
        alpha = row['alpha']
        covariance = independent_covariance(cfg['data'], alpha)
        close(row['covariance'], covariance, name+' covariance')
        close(np.diag(covariance), np.ones(len(covariance)), name+' marginal source variance')
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise AssertionError('Nonpositive source covariance')
        eps = draw(covariance, cfg['n_train'], data_seed+100)
        val_eps = draw(covariance, cfg['n_val'], data_seed+110)
        velocity = datasets['train']['target']-eps
        independent_pca_check(velocity, row['pca'], f'{data_seed} {name} velocity')
        compare_mapping(independent_spectrum(datasets['val']['target']-val_eps, row['pca']),
                        row['validation_velocity'], name+' validation velocity spectrum')
        compare_mapping(independent_spectrum(val_eps, train_pca(eps)), row['validation_source'],
                        name+' validation source spectrum', atol=1e-7, rtol=1e-7)
        expected_floor_count = int(np.sum(np.asarray(row['pca']['eigenvalues']) < row['pca']['floor']))
        if row['floored_directions'] != expected_floor_count:
            raise AssertionError('PCA floored direction count differs')
    sources = geometry['sources']; target = geometry['validation_target']
    gate = (sources['gp']['validation_velocity']['top2_energy'] > .9
            and sources['white']['validation_velocity']['top2_energy'] < .9
            and target['mean_within_patch_std'] > .02 and target['roughness'] > .002)
    if not gate or not geometry['gate_pass']:
        raise AssertionError('Pretraining geometry gate failed')
    return datasets


def pairs(cfg, ds, alpha, data_seed, split):
    offset = {'val': 800, 'test': 900}[split]
    eps = draw(independent_covariance(cfg['data'], alpha), len(ds['target']), data_seed+offset)
    times = np.random.default_rng(data_seed+offset+1).uniform(0, 1, len(eps))
    return (1-times[:, None])*eps+times[:, None]*ds['target'], ds['target']-eps, times, eps


def pca_object(value):
    from fm_stress.metrics import PCA
    return PCA(np.asarray(value['mean']), np.asarray(value['components']),
               np.asarray(value['eigenvalues']), value['floor'])


def validate_run(path, cfg, datasets, geometry, fingerprint, expected_device, replay_euler):
    from fm_stress.prior_mixture import make_model
    record = load(path)
    dest = path.parent.parent
    name, alpha, head = record['run'], record['alpha'], record['head']
    data_seed, seed = record['data_seed'], record['seed']
    expected_name = 'white' if alpha == 0 else 'gp' if alpha == 1 else f'mix{round(alpha*100):03d}'
    if name != f'{expected_name}_{head}_seed{seed}' or record['source'] != expected_name:
        raise AssertionError('Run identity differs '+name)
    if record['fingerprint'] != fingerprint or record['config_sha256'] != stable_hash(cfg):
        raise AssertionError('Run provenance differs '+name)
    if record['status'] != 'complete' or record['steps'] != cfg['steps']:
        raise AssertionError('Run completeness/budget differs '+name)
    if record['device'] != expected_device:
        raise AssertionError('Run device differs '+name)
    curve = record['curve']
    schedule = list(range(cfg['eval_every'], cfg['steps']+1, cfg['eval_every']))
    if not schedule or schedule[-1] != cfg['steps']:
        schedule.append(cfg['steps'])
    if [row['step'] for row in curve] != schedule or curve != load(dest/'training'/f'{name}.json'):
        raise AssertionError('Validation schedule/log differs '+name)
    for row in curve:
        fraction = (row['step']-1)/max(cfg['steps']-1, 1)
        lr = cfg['final_learning_rate']+.5*(cfg['learning_rate']-cfg['final_learning_rate'])*(1+np.cos(np.pi*fraction))
        close(row['learning_rate'], lr, name+' learning-rate schedule')
    best = min(curve, key=lambda row: row['validation_objective'])
    if record['best_step'] != best['step']:
        raise AssertionError('Checkpoint was not selected by validation '+name)
    close(record['best_validation_objective'], best['validation_objective'], name+' best objective')
    cp, sp = dest/'checkpoints'/f'{name}.pt', dest/'samples'/f'{name}.npz'
    if sha(cp) != record['checkpoint_sha256'] or sha(sp) != record['samples_sha256']:
        raise AssertionError('Checkpoint/sample content hash differs '+name)
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    for key, value in dict(config=cfg, data_seed=data_seed, alpha=alpha, head=head, seed=seed,
                           step=best['step'], fingerprint=fingerprint).items():
        if saved[key] != value:
            raise AssertionError('Checkpoint metadata differs '+key+' '+name)
    pca = geometry['sources'][expected_name]['pca']
    if saved['pca'] != pca:
        raise AssertionError('Checkpoint PCA differs from train geometry '+name)
    model = make_model(cfg, head, pca_object(pca), seed, torch.device('cpu'))
    initial_encoder = np.concatenate([p.detach().numpy().ravel() for p in model.base.encoder.parameters()])
    if hashlib.sha256(np.ascontiguousarray(initial_encoder).tobytes()).hexdigest() != record['encoder_initial_sha256']:
        raise AssertionError('Initial encoder cannot be reconstructed '+name)
    model.load_state_dict(saved['state_dict'], strict=True); model.eval()
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != record['parameters']:
        raise AssertionError('Parameter count differs '+name)
    close(model.q.numpy(), np.asarray(pca['components']).T, name+' PCA buffer', atol=1e-7)
    close(model.scales.numpy(), np.sqrt(np.maximum(pca['eigenvalues'], pca['floor'])), name+' scale buffer', atol=1e-6)
    vx, vu, vt, _ = pairs(cfg, datasets['val'], alpha, data_seed, 'val')
    val_prediction = replay(model, vx, vt, datasets['val']['context'])
    # Training validation computes against float32 labels, unlike exported raw
    # test scores which intentionally retain the float64 independent coupling.
    error = val_prediction-vu.astype(np.float32)
    if head in ('whitened_mlp', 'balanced_mlp'):
        transformed = error @ np.asarray(pca['components']).T / np.sqrt(np.maximum(pca['eigenvalues'], pca['floor']))
    else:
        transformed = error
    validation_objective = float(np.square(transformed).mean())
    close(validation_objective, best['validation_objective'], name+' CPU validation objective', atol=3e-5, rtol=4e-3)
    close(np.square(error).mean(), best['validation_raw_mse'], name+' CPU raw validation objective', atol=3e-5, rtol=4e-3)
    tx, tu, tt, eps = pairs(cfg, datasets['test'], alpha, data_seed, 'test')
    arrays = np.load(sp)
    for key, value in dict(x=tx, velocity=tu, time=tt, source=eps,
                           history=datasets['test']['context'], target=datasets['test']['target']).items():
        close(arrays[key], value, name+' saved '+key)
    prediction = arrays['prediction']
    cpu_prediction = replay(model, tx, tt, datasets['test']['context'])
    close(cpu_prediction, prediction, name+' CPU test velocity replay', atol=5e-4, rtol=4e-3)
    compare_mapping(independent_velocity(prediction, tu, pca), record['metrics'], name+' velocity scores')
    compare_mapping(independent_velocity(prediction, tu, geometry['target_pca']),
                    record['fixed_target_basis_metrics'], name+' fixed-target PCA scores')
    compare_mapping(independent_spectrum(tu, pca), record['test_spectrum'], name+' held-out velocity spectrum')
    n = min(cfg['n_paths'], cfg['n_test'])
    if arrays['generated'].shape != (n, cfg['data']['horizon']):
        raise AssertionError('Single-path forecast shape differs '+name)
    generated_roughness = float(np.abs(np.diff(arrays['generated'].astype(float), axis=1)).mean())
    target_roughness = float(np.abs(np.diff(datasets['test']['target'][:n], axis=1)).mean())
    close(record['metrics']['generated_roughness'], generated_roughness, name+' generated roughness')
    close(record['metrics']['target_roughness'], target_roughness, name+' target roughness')
    metrics = record['metrics']
    for key, pc1, residual in [('success', metrics['pc1_mse'], metrics['residual_mse']),
                              ('normalized_success_diagnostic', metrics['pc1_normalized_mse'], metrics['residual_normalized_mse'])]:
        compare_mapping(independent_success(pc1, residual, generated_roughness, target_roughness), metrics[key], name+' '+key)
    nc, ns = min(cfg['n_ensemble_contexts'], cfg['n_test']), cfg['ensemble_size']
    es = draw(independent_covariance(cfg['data'], alpha), nc*ns, data_seed+1000).reshape(nc, ns, -1)
    close(arrays['ensemble_source'], es, name+' ensemble source')
    close(arrays['ensemble_history'], datasets['test']['context'][:nc], name+' ensemble histories')
    close(arrays['ensemble_target'], datasets['test']['target'][:nc], name+' ensemble target')
    scores = independent_scores(arrays['ensemble'], datasets['test']['target'][:nc])
    compare_mapping(scores, record['per_context'], name+' per-context probabilistic scores')
    compare_mapping({k: float(v.mean()) for k, v in scores.items()}, record['forecast_means'], name+' probabilistic means')
    solver = record['solver_sensitivity']
    if seed == cfg['seeds'][0]:
        ni = min(8, nc)
        if solver['n_contexts'] != ni or solver['euler_steps'] != 2*cfg['euler_steps']:
            raise AssertionError('Solver refinement scope differs '+name)
        base = {k: v[:ni] for k, v in scores.items()}
        refined = independent_scores(arrays['generated128'], datasets['test']['target'][:ni])
        close(solver['endpoint_rmse'], np.sqrt(np.square(arrays['generated128'].astype(float)-arrays['ensemble'][:ni]).mean()), name+' solver endpoint')
        compare_mapping({k: float(refined[k].mean()-base[k].mean()) for k in base}, solver['score_changes'], name+' solver score changes')
    elif solver or 'generated128' in arrays.files:
        raise AssertionError('Unexpected refinement outside designated seed '+name)
    gradients = record['gradient_diagnostics']
    expected_gradient_steps = [step for step in cfg['gradient_steps'] if step <= cfg['steps']]
    if [row['step'] for row in gradients[:-1]] != expected_gradient_steps:
        raise AssertionError('Gradient checkpoint coverage differs '+name)
    selected = gradients[-1]
    if selected['step'] != best['step'] or not selected['selected_checkpoint']:
        raise AssertionError('Final gradient probe is not selected checkpoint '+name)
    for diagnostic in gradients:
        close(sum(row['raw_mse_contribution'] for row in diagnostic['groups'].values()),
              diagnostic['raw_total_mse'], name+' gradient loss decomposition', atol=2e-5, rtol=2e-5)
        for group in diagnostic['groups'].values():
            close(group['gradient_norm_per_component']*group['components'], group['gradient_norm'], name+' per-component gradient norm')
        for cosine in diagnostic['gradient_cosines'].values():
            if cosine is not None and not -1 <= cosine <= 1:
                raise AssertionError('Invalid gradient cosine '+name)
    npb = min(cfg['gradient_probe_size'], len(vu))
    probe_error = error[:npb]
    close(selected['raw_total_mse'], np.square(probe_error).mean(), name+' selected gradient-probe raw loss', atol=4e-5, rtol=5e-3)
    replay_error = None
    if replay_euler:
        # One context, up to four common draws per fit: independent solver code,
        # deliberately bounded; full-array scoring above covers every saved path.
        n_draws = min(4, ns)
        generated_cpu = independent_euler(model, es[0, :n_draws],
                                         np.repeat(datasets['test']['context'][:1], n_draws, axis=0), cfg['euler_steps'])
        replay_error = float(np.max(np.abs(generated_cpu-arrays['ensemble'][0, :n_draws])))
        close(generated_cpu, arrays['ensemble'][0, :n_draws], name+' CPU Euler endpoint replay', atol=2e-3, rtol=8e-3)
    return dict(name=name, data_seed=data_seed, source=record['source'], head=head, seed=seed,
                validation_objective_cpu=validation_objective,
                validation_objective_recorded=best['validation_objective'],
                test_velocity_max_abs_difference=float(np.max(np.abs(cpu_prediction-prediction))),
                bounded_euler_max_abs_difference=replay_error,
                encoder_initial_sha256=record['encoder_initial_sha256'], batch_hashes=record['batch_hashes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default='results/prior-mixture')
    parser.add_argument('--config', default='configs/prior_mixture.json')
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'mps'], default='mps', help='Required device in saved run records')
    parser.add_argument('--skip-euler-replay', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    root, cfg = Path(args.results), load(args.config)
    torch.set_num_threads(2)
    manifest = load(root/'manifest.json')
    fingerprint = manifest['fingerprint']
    if manifest['config_sha256'] != stable_hash(cfg) or stable_hash({k: v for k, v in manifest.items() if k != 'fingerprint'}) != fingerprint:
        raise AssertionError('Manifest/config fingerprint differs')
    if load(root/'config.json') != cfg:
        raise AssertionError('Saved config differs')
    for name, digest in manifest['source_hashes'].items():
        if sha(ROOT/name) != digest:
            raise AssertionError('Frozen numerical source changed '+name)
    expected = {(int(d), float(a), h, int(s)) for d in cfg['data_seeds'] for a in cfg['alphas']
                for h in cfg['heads']+(cfg.get('gp_control_heads', []) if a == 1 else []) for s in cfg['seeds']}
    observed, records, record_hashes, geometry_hashes = set(), [], {}, {}
    for data_seed in cfg['data_seeds']:
        dest = root/f'dataset_{data_seed}'
        geometry = load(dest/'geometry.json')
        if geometry['fingerprint'] != fingerprint or geometry['config_sha256'] != stable_hash(cfg):
            raise AssertionError('Geometry provenance differs')
        datasets = validate_geometry(cfg, data_seed, geometry)
        geometry_hashes[str((dest/'geometry.json').relative_to(root))] = sha(dest/'geometry.json')
        for path in sorted((dest/'runs').glob('*.json')):
            record = load(path)
            identity = (record['data_seed'], record['alpha'], record['head'], record['seed'])
            if identity not in expected or identity in observed or record['data_seed'] != data_seed:
                raise AssertionError('Unexpected/duplicate run identity '+str(path))
            observed.add(identity)
            result = validate_run(path, cfg, datasets, geometry, fingerprint, args.device, not args.skip_euler_replay)
            records.append(result)
            record_hashes[str(path.relative_to(root))] = sha(path)
            print('VERIFIED', data_seed, record['run'], flush=True)
    for seed in cfg['seeds']:
        paired = [r for r in records if r['seed'] == seed]
        if len({r['encoder_initial_sha256'] for r in paired}) > 1:
            raise AssertionError('Encoder initialization differs across paired cells')
        if len({json.dumps(r['batch_hashes'], sort_keys=True) for r in paired}) > 1:
            raise AssertionError('First/last common random training draws differ across paired cells')
    complete = observed == expected and len(records) == len(expected)
    if not complete and not args.allow_partial:
        raise AssertionError(f'Incomplete suite: {len(records)} of {len(expected)} runs')
    summary = dict(complete=complete, expected_runs=len(expected), verified_runs=len(records),
                   manifest_fingerprint=fingerprint, config_sha256=stable_hash(cfg),
                   record_sha256=record_hashes, geometry_sha256=geometry_hashes,
                   required_device=args.device, bounded_euler_replay=not args.skip_euler_replay,
                   scope='Independent data/PCA/metric reconstruction; full CPU validation and test velocity replay; bounded CPU Euler replay. Training trajectories not independently rerun.',
                   missing=sorted(expected-observed), runs=records)
    output = Path(args.output) if args.output else root/'independent_verification.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('runs', 'record_sha256', 'geometry_sha256', 'missing')}, indent=2))


if __name__ == '__main__':
    main()
