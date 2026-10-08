"""Validation-selected MultiTask Elastic Net and train-residual forecasts.

Uses the extension's already TRAIN-channel-standardized arrays without another
normalization. Each flattened history lag/channel is selected jointly across all
future horizon/channel outputs. This is an external forecasting baseline, not FM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import MultiTaskElasticNet
from threadpoolctl import threadpool_limits

from .experiment import save_json
from .extended_data import build_data
from .extended_experiment import evaluation_indices
from .extended_metrics import forecast_scores, means

ALPHAS = (.001, .01, .1, 1.)
L1_RATIOS = (.5, .9, 1.)


@threadpool_limits.wrap(limits=4)
def select_elastic_net(x, y, vx, vy, *, max_iter=50000, tol=1e-5):
    """Fit the prespecified 12 candidates on TRAIN; select by validation MSE.

    Nonconverged candidates remain in the audit record but cannot win. This rule
    is fixed before test evaluation. An entirely nonconverged grid is an error.
    """
    candidates, fitted = [], []
    for alpha in ALPHAS:
        for ratio in L1_RATIOS:
            model = MultiTaskElasticNet(alpha=alpha, l1_ratio=ratio,
                fit_intercept=True, max_iter=max_iter, tol=tol, selection='cyclic')
            start = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(x, y)
            messages = [str(w.message) for w in caught]
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            validation = float(np.mean((model.predict(vx)-vy)**2))
            if not np.isfinite(validation) or not np.isfinite(model.coef_).all():
                raise FloatingPointError('Nonfinite Elastic Net fit')
            candidates.append({'alpha':alpha, 'l1_ratio':ratio,
                'validation_mse':validation, 'converged':converged,
                'n_iter':int(model.n_iter_), 'dual_gap':float(model.dual_gap_),
                'warnings':messages, 'seconds':time.perf_counter()-start,
                'active_input_features':int(np.any(model.coef_ != 0, axis=0).sum())})
            fitted.append(model)
    eligible = [i for i,r in enumerate(candidates) if r['converged']]
    if not eligible:
        raise RuntimeError('No Elastic Net candidate converged; increase max_iter before using a fresh output directory')
    selected = min(eligible, key=lambda i:(candidates[i]['validation_mse'],
                                         candidates[i]['alpha'], candidates[i]['l1_ratio']))
    return fitted[selected], candidates, selected


def fit_sparse_baselines(bundle, cfg, out):
    """Write sparse_baselines.json and sparse_baseline_samples.npz; return record.

    Uses exactly extension evaluation_indices and evaluation_seed + 15 residual
    draw indices. A residual draw is an entire F*C vector; time/channel dependence
    is preserved. Training residuals are in sample and may understate uncertainty.
    """
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if tuple(cfg.get('elastic_alphas',ALPHAS)) != ALPHAS or tuple(cfg.get('elastic_l1_ratios',L1_RATIOS)) != L1_RATIOS:
        raise ValueError('Elastic Net candidate grid differs from the prespecified 12 candidates')
    relevant_cfg = {k:cfg[k] for k in ('n_ensemble_contexts','ensemble_size','evaluation_seed')}
    relevant_cfg.update(alphas=list(ALPHAS),l1_ratios=list(L1_RATIOS),
        max_iter=int(cfg.get('elastic_net_max_iter',50000)),tol=float(cfg.get('elastic_net_tol',1e-5)))
    source_root=Path(__file__).parent
    provenance={'source_sha256':{p:hashlib.sha256((source_root/p).read_bytes()).hexdigest()
        for p in ('sparse_baselines.py','extended_data.py','extended_metrics.py','extended_experiment.py')},
        'config':relevant_cfg, 'sklearn_version':sklearn.__version__,'numpy_version':np.__version__,
        'metadata_sha256':hashlib.sha256(json.dumps(bundle.metadata,sort_keys=True).encode()).hexdigest()}
    fingerprint=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    record_path,samples_path=out/'sparse_baselines.json',out/'sparse_baseline_samples.npz'
    if record_path.exists():
        previous=json.loads(record_path.read_text())
        if previous.get('fingerprint') != fingerprint or previous.get('provenance') != provenance:
            raise RuntimeError('Sparse baseline provenance changed; use a fresh output directory')
        if not samples_path.exists() or hashlib.sha256(samples_path.read_bytes()).hexdigest()!=previous.get('samples_sha256'):
            raise RuntimeError('Sparse baseline cached samples are missing or changed')
        print('SKIP Elastic Net',out.name,'verified cache',flush=True)
        return previous
    if samples_path.exists():
        raise RuntimeError('Sparse baseline samples exist without a complete record; use a fresh output directory')
    train, val, test = [bundle.splits[s] for s in ('train','val','test')]
    x = np.asfortranarray(train.context.reshape(len(train), -1), dtype=np.float64)
    y = np.asfortranarray(train.target.reshape(len(train), -1), dtype=np.float64)
    vx = val.context.reshape(len(val), -1)
    vy = val.target.reshape(len(val), -1)
    max_iter, tol = int(cfg.get('elastic_net_max_iter',50000)), float(cfg.get('elastic_net_tol',1e-5))
    model, candidates, selected = select_elastic_net(x,y,vx,vy,max_iter=max_iter,tol=tol)
    ix = evaluation_indices(test,cfg['n_ensemble_contexts'])
    target = test.target[ix]
    n,f,c = target.shape
    s = int(cfg['ensemble_size'])
    point = model.predict(test.context[ix].reshape(n,-1)).reshape(n,f,c)
    residuals = y-model.predict(x)
    draws = np.random.default_rng(cfg['evaluation_seed']+15).integers(len(y),size=(n,s))
    generated = point[:,None]+residuals[draws].reshape(n,s,f,c)
    records = {}
    for name, samples, kind in (
        ('multitask_elastic_net', np.repeat(point[:,None],s,axis=1),
         'deterministic point forecast / degenerate predictive distribution'),
        ('multitask_elastic_net_residual_bootstrap', generated,
         'TRAIN in-sample whole-vector residual bootstrap; may understate uncertainty')):
        scores = forecast_scores(samples,target)
        records[name] = {'means':means(scores), 'per_context':{k:v.tolist() for k,v in scores.items()}, 'type':kind}
    mask = np.any(model.coef_ != 0,axis=0)
    record = {'method':'sklearn.linear_model.MultiTaskElasticNet',
        'status':'complete','fingerprint':fingerprint,'provenance':provenance,
        'sklearn_version':sklearn.__version__, 'selected_candidate_index':selected,
        'selected_alpha':candidates[selected]['alpha'],
        'selected_l1_ratio':candidates[selected]['l1_ratio'],
        'validation_mse':candidates[selected]['validation_mse'],
        'selection':'Minimum validation MSE among converged candidates; no test selection; no train+validation refit',
        'normalization':'Existing build_data TRAIN-only global channel scaler; no further feature scaling',
        'scaler_mean':bundle.metadata['scaler_mean'], 'scaler_scale':bundle.metadata['scaler_scale'],
        'flattening':'C order, time first/channel last, for history and future',
        'active_input_features':int(mask.sum()), 'total_input_features':len(mask),
        'active_history_lag_channel_indices':np.argwhere(mask.reshape(train.context.shape[1:])).tolist(),
        'coefficient_orientation':'Saved coefficients: [flattened future, flattened history]; feature groups are columns',
        'max_iter':max_iter,'tol':tol, 'candidates':candidates,'records':records,
        'training_context_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        'training_target_sha256':hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
        'evaluation_indices':ix.tolist(), 'residual_draw_seed':cfg['evaluation_seed']+15,
        'uncertainty_caveat':'Residuals are in-sample TRAIN errors; bootstrap may understate uncertainty and assumes transferable residuals.'}
    np.savez_compressed(samples_path,indices=ix,
        coefficients=model.coef_,intercept=model.intercept_,active_input_mask=mask,
        residual_draw_indices=draws,training_residuals=residuals,
        multitask_elastic_net=point,multitask_elastic_net_residual_bootstrap=generated)
    record['samples_sha256']=hashlib.sha256(samples_path.read_bytes()).hexdigest()
    record['seconds']=time.perf_counter()-started
    save_json(record_path,record)
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/extension.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--dataset',action='append')
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    for name,dc in cfg['datasets'].items():
        if args.dataset and name not in args.dataset:
            continue
        print('START Elastic Net',name,flush=True)
        result=fit_sparse_baselines(build_data(dc),cfg,Path(args.output)/name)
        print('DONE Elastic Net',name,'alpha',result['selected_alpha'],
              'l1_ratio',result['selected_l1_ratio'],'active',result['active_input_features'],
              'CRPS',result['records']['multitask_elastic_net_residual_bootstrap']['means']['fair_crps'],flush=True)


if __name__ == '__main__':
    main()
