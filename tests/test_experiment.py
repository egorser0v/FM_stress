"""Runner-level invariants: coordinate correctness and actual paired training.

The end-to-end test intercepts the real runner before/at its first update; it
therefore catches architecture-dependent RNG consumption or encoder resets.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from fm_stress.experiment import (CoordinateModel, fixed_pairs, generate,
                                  get_data, train_cell)
from fm_stress.metrics import PCA, fit_pca
from fm_stress.data import sample_source, source_covariance
from fm_stress import models


class FixedCoordinateField(nn.Module):
    def __init__(self, coordinate_velocity):
        super().__init__()
        self.horizon = len(coordinate_velocity)
        self.register_buffer("velocity", torch.tensor(coordinate_velocity, dtype=torch.float32))

    def forward(self, x, t, history):
        return self.velocity.expand_as(x)


def tiny_config():
    return {
        "data": {"lookback": 8, "horizon": 4, "lengthscale": 2.0,
                 "variance": 1.0, "nugget": 1e-4},
        "data_seed": 170,
        "n_train": 32, "n_val": 12, "n_test": 12, "n_paths": 4,
        "batch_size": 8, "steps": 1, "eval_every": 1, "euler_steps": 2,
        "learning_rate": 1e-3, "weight_decay": 1e-4, "clip_grad": 1.0,
        "model": {"width": 8, "encoder_dim": 4, "depth": 3,
                  "state_size": 4, "time_dim": 8},
    }


class TestCoordinates(unittest.TestCase):
    def test_whitening_is_invertible_and_derivatives_have_no_mean_offset(self):
        # Nonzero PCA mean exposes accidentally centering derivatives. The
        # independent expected transform uses numpy, not CoordinateModel helpers.
        q, _ = np.linalg.qr(np.random.default_rng(12).normal(size=(4, 4)))
        pca = PCA(np.array([100., -200., 70., 2.]), q.T,
                  np.array([16., 4., 1., .25]), 1e-8)
        raw_velocity = np.array([.3, -.4, .2, .5])
        expected_w = q / np.sqrt(pca.eigenvalues)[None]
        model = CoordinateModel(FixedCoordinateField(raw_velocity @ expected_w), pca)
        x = torch.tensor([[1., 2., 3., 4.], [-2., .2, .8, 1.]], dtype=torch.float32)
        z = model.to_coordinates(x)
        np.testing.assert_allclose(z.numpy(), x.numpy() @ expected_w, atol=2e-6)
        torch.testing.assert_close(model.from_coordinates(z), x, atol=2e-6, rtol=2e-6)
        h, t = torch.zeros((2, 8)), torch.zeros(2)
        np.testing.assert_allclose(model(x, t, h).numpy(),
                                   np.broadcast_to(raw_velocity, (2, 4)), atol=2e-6)
        for steps in (1, 7):
            np.testing.assert_allclose(generate(model, x, h, steps),
                                       x.numpy() + raw_velocity, atol=3e-6)
        # A linear change of coordinates must commute with interpolation.
        y = -2 * x
        times = torch.tensor([[.2], [.8]])
        torch.testing.assert_close(model.to_coordinates((1-times)*x + times*y),
                                   (1-times)*model.to_coordinates(x) + times*model.to_coordinates(y))


class TestExperimentPairing(unittest.TestCase):
    def test_splits_and_fixed_evaluation_pairs(self):
        cfg = tiny_config()
        gp, data = get_data(cfg)
        _, again = get_data(cfg)
        for split in data:
            np.testing.assert_array_equal(data[split].target, again[split].target)
        self.assertFalse(np.array_equal(data["val"].target, data["test"].target))
        first = fixed_pairs(cfg, gp, data["test"], "gp", "test")
        for actual, expected in zip(fixed_pairs(cfg, gp, data["test"], "gp", "test"), first):
            np.testing.assert_array_equal(actual, expected)
        x, u, t, eps = first
        np.testing.assert_allclose(u, data["test"].target-eps)
        np.testing.assert_allclose(x, (1-t[:, None])*eps+t[:, None]*data["test"].target)
        white = fixed_pairs(cfg, gp, data["test"], "white", "test")
        np.testing.assert_array_equal(white[2], t)
        recovered_standard_normals = np.linalg.solve(np.linalg.cholesky(source_covariance(gp, "gp")), eps.T).T
        np.testing.assert_allclose(recovered_standard_normals, white[3], atol=2e-12)
        self.assertFalse(np.array_equal(fixed_pairs(cfg, gp, data["val"], "gp", "val")[3], eps))

    def test_actual_runner_matches_encoder_initialization_and_training_draws(self):
        cfg = tiny_config()
        gp, data = get_data(cfg)
        pcas = {source: fit_pca(data["train"].target - sample_source(gp, cfg["n_train"], source, 19))
                for source in ("white", "gp")}
        encoder_states, initial_batches = {}, {}
        active = [None]
        original_groups = models.optimizer_groups
        original_forward = CoordinateModel.forward_coordinates

        def capture_groups(model, *args, **kwargs):
            encoder_states[active[0]] = {key: value.detach().clone()
                                        for key, value in model.base.encoder.state_dict().items()}
            return original_groups(model, *args, **kwargs)

        def capture_forward(model, x, t, history):
            if model.training and active[0] not in initial_batches:
                initial_batches[active[0]] = tuple(z.detach().clone() for z in (x, t, history))
            return original_forward(model, x, t, history)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(models, "optimizer_groups", capture_groups), \
                patch.object(CoordinateModel, "forward_coordinates", capture_forward):
            for source, kind in (("white", "mlp"), ("white", "s4"),
                                 ("gp", "mlp"), ("gp", "whitened_mlp")):
                active[0] = (source, kind)
                result = train_cell(cfg, gp, data, pcas, source, kind, 3, tmp, torch.device("cpu"))
                self.assertEqual(result["best_step"], 1)
                self.assertTrue(np.isfinite(result["metrics"]["velocity_mse"]))
        baseline = encoder_states[("white", "mlp")]
        for state in encoder_states.values():
            self.assertEqual(set(state), set(baseline))
            for name in baseline:
                torch.testing.assert_close(state[name], baseline[name], atol=0, rtol=0)
        for left, right in ((("white", "mlp"), ("white", "s4")),
                            (("gp", "mlp"), ("gp", "whitened_mlp"))):
            for a, b in zip(initial_batches[left], initial_batches[right]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        # Times and history examples are paired even across distinct sources.
        for index in (1, 2):
            torch.testing.assert_close(initial_batches[("white", "mlp")][index],
                                       initial_batches[("gp", "mlp")][index], atol=0, rtol=0)

    def test_pilot_and_primary_do_not_share_synthetic_test_draws(self):
        root = Path(__file__).resolve().parents[1]
        configs = [json.loads((root / "configs" / filename).read_text())
                   for filename in ("pilot.json", "main.json")]
        pilot_seeds = {configs[0]["data_seed"] + i for i in range(3)}
        main_seeds = {configs[1]["data_seed"] + i for i in range(3)}
        self.assertFalse(pilot_seeds & main_seeds,
                         "Pilot and scored runs need separate data draws if pilot informs choices")


if __name__ == "__main__":
    unittest.main()
