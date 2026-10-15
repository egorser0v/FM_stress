import json
from pathlib import Path

import numpy as np
import pytest

import fm_stress.etth1_normalization as e4
from fm_stress.extended_data import WindowDataset, build_data

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / 'data/real/ETTh1.csv').exists(), reason='ETTh1.csv not fetched')


def test_history_normalization_is_level_equivariant():
    rng = np.random.default_rng(0)
    ds = WindowDataset(rng.standard_normal((5, 48, 1)), rng.standard_normal((5, 24, 1)), np.arange(5))
    shifted = WindowDataset(ds.context + 3.0, ds.target + 3.0, ds.starts)
    a, mu, sd = e4.history_normalize(ds)
    b, mu2, sd2 = e4.history_normalize(shifted)
    assert np.allclose(a.context, b.context) and np.allclose(a.target, b.target)
    assert np.allclose(mu2 - mu, 3.0) and np.allclose(sd2, sd)
    assert np.allclose(mu + sd * a.target, ds.target)


def test_cpu_smoke_both_normalizations(tmp_path):
    import torch
    cfg = json.loads((ROOT / 'configs/etth1_normalization.json').read_text())
    cfg = {**cfg, 'steps': 3, 'eval_every': 3, 'n_ensemble_contexts': 4, 'ensemble_size': 2, 'euler_steps': 2}
    bundle = build_data(cfg['dataset'])
    fp = e4.prepare_root(cfg, tmp_path)
    for norm in ('global', 'history'):
        rec = e4.train_run(cfg, bundle, tmp_path, norm, 'mlp', 'gp', 31, fp, torch.device('cpu'))
        assert np.isfinite(rec['forecast_means']['fair_crps'])
