import numpy as np
import pytest

from scripts.analyze_prior_oracle import error_decomposition


def test_empirical_cross_term_preserved_with_rotated_axes():
    rng = np.random.default_rng(57)
    label = rng.normal(size=(17, 5))
    oracle = rng.normal(size=(17, 5))
    # Deliberately correlated discrepancies make dropping the cross term wrong.
    prediction = oracle + .4 * (oracle - label)
    q, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    values = error_decomposition(prediction, oracle, label, q)
    oracle_mse = values["oracle_label_mse"]["overall_mean"]
    assert values["signed_cross_term"]["overall_mean"] == pytest.approx(.8 * oracle_mse)
    assert values["model_oracle_mse"]["overall_mean"] == pytest.approx(.16 * oracle_mse)
    assert values["model_label_mse"]["overall_mean"] == pytest.approx(1.96 * oracle_mse)
    assert values["identity_max_absolute_error"] < 1e-13


def test_oracle_prediction_has_zero_discrepancy_and_cross_term():
    oracle = np.arange(28).reshape(7, 4) / 13
    values = error_decomposition(oracle, oracle, np.zeros((7, 4)), np.eye(4))
    assert values["model_oracle_mse"]["overall_mean"] == 0
    assert values["signed_cross_term"]["overall_mean"] == 0
    assert values["model_label_mse"] == values["oracle_label_mse"]
