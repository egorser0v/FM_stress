"""Synthetic joint GP windows with leakage-free Sundial instance normalization.

All random streams are local numpy Generators. Float64 is used for geometry and
linear algebra; callers may convert training tensors to float32 on MPS.
"""
from dataclasses import dataclass, asdict
from typing import Union
import numpy as np


@dataclass(frozen=True)
class GPConfig:
    lookback: int = 32
    horizon: int = 16
    lengthscale: float = 8.0
    variance: float = 1.0
    nugget: float = 1e-6
    normalization_threshold: float = 1e-2

    def __post_init__(self):
        if self.lookback < 2 or self.horizon < 3:
            raise ValueError("lookback >= 2 and horizon >= 3 are required")
        if self.lengthscale <= 0 or self.variance <= 0 or self.nugget < 0:
            raise ValueError("Invalid kernel parameters")
        if self.normalization_threshold <= 0:
            raise ValueError("Normalization threshold must be positive")

    def to_dict(self):
        return asdict(self)


@dataclass
class GPDataset:
    context: np.ndarray
    target: np.ndarray
    raw_context: np.ndarray
    raw_target: np.ndarray
    location: np.ndarray
    scale: np.ndarray

    def __len__(self):
        return len(self.target)

    def unnormalize(self, normalized):
        """Accept [N,F] or [N,S,F] arrays (one normalization per history)."""
        normalized = np.asarray(normalized)
        shape = (len(self),) + (1,) * (normalized.ndim - 1)
        return normalized * self.scale.reshape(shape) + self.location.reshape(shape)


def _rng(seed: Union[int, np.random.Generator]):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def rbf_covariance(size, lengthscale, variance=1.0, nugget=1e-6):
    """Unit-spaced stationary squared-exponential covariance plus tiny nugget."""
    if size < 1 or lengthscale <= 0 or variance <= 0 or nugget < 0:
        raise ValueError("Invalid covariance parameters")
    grid = np.arange(size, dtype=np.float64)
    return variance * np.exp(-0.5 * ((grid[:, None] - grid) / lengthscale) ** 2) + nugget * np.eye(size)


def kernel_covariance(config, size=None):
    return rbf_covariance(config.lookback + config.horizon if size is None else size,
                          config.lengthscale, config.variance, config.nugget)


def sundial_normalize(context, target=None, threshold=1e-2):
    """Official revin=True behavior: population std, <=.01 replaced by 1.

    Reference: thuml/sundial-base-128m/modeling_sundial.py, forward(),
    lines 397--418 at retrieval 2026-10-04. Statistics use ONLY the history.
    """
    context = np.asarray(context, dtype=np.float64)
    if context.ndim != 2 or not np.isfinite(context).all():
        raise ValueError("context must be a finite [N,L] array")
    location = context.mean(axis=1, keepdims=True)
    scale = context.std(axis=1, keepdims=True, ddof=0)
    scale = np.where(scale > threshold, scale, 1.0)
    normalized_context = (context - location) / scale
    normalized_target = None if target is None else (np.asarray(target) - location) / scale
    return normalized_context, normalized_target, location, scale


def sample_dataset(config: GPConfig, n: int, seed=0):
    """Draw each lookback and future JOINTLY; windows themselves independent."""
    if n <= 0:
        raise ValueError("n must be positive")
    chol = np.linalg.cholesky(kernel_covariance(config))
    windows = _rng(seed).standard_normal((n, config.lookback + config.horizon)) @ chol.T
    raw_context, raw_target = windows[:, :config.lookback], windows[:, config.lookback:]
    context, target, location, scale = sundial_normalize(raw_context, raw_target, config.normalization_threshold)
    return GPDataset(context, target, raw_context, raw_target, location, scale)


def source_covariance(config: GPConfig, source: str):
    if source in ("white", "white_noise"):
        return np.eye(config.horizon)
    if source in ("gp", "matched_gp"):
        return kernel_covariance(config, config.horizon)
    raise ValueError("source must be 'white' or 'gp'")


def sample_source(config: GPConfig, n: int, source: str, seed=0):
    """Source drawn in normalized observation coordinates, independently of y.

    Matched means same RBF kernel/lengthscale as the RAW data generator. Applying
    each target's history scale to its source would break unconditional source--
    target independence, so we deliberately do not do that.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    return _rng(seed).standard_normal((n, config.horizon)) @ np.linalg.cholesky(source_covariance(config, source)).T


def interpolate(target, source, time):
    target, source = np.asarray(target), np.asarray(source)
    time = np.asarray(time)
    if time.ndim == 1:
        time = time[:, None]
    if target.shape != source.shape or np.any((time < 0) | (time > 1)):
        raise ValueError("Matched array shapes and time in [0,1] are required")
    return (1.0 - time) * source + time * target, target - source


def conditional_target(config: GPConfig, dataset: GPDataset):
    """Exact normalized-future Gaussian conditioned on the FULL RAW history.

    This is privileged information: the network only sees normalized history.
    Therefore the resulting oracle is a lower-bound diagnostic, not an attainable
    Bayes oracle for the network's information set.
    """
    k = kernel_covariance(config)
    l = config.lookback
    a = np.linalg.solve(k[:l, :l], k[:l, l:])
    means = (dataset.raw_context @ a - dataset.location) / dataset.scale
    residual_cov = k[l:, l:] - k[l:, :l] @ a
    residual_cov = (residual_cov + residual_cov.T) / 2
    covs = residual_cov[None, :, :] / dataset.scale[:, :, None] ** 2
    return means, covs


def conditional_oracle_velocity(x, time, target_mean, target_cov, source_cov):
    """Analytic independent-Gaussian CFM regression E[y-epsilon | x_t, raw h].

    Uses linear solves without adding artificial jitter to the metric. Kernel
    nugget ensures invertibility, including endpoints t=0 and t=1.
    """
    x = np.asarray(x, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    if time.ndim == 0:
        time = np.full((len(x), 1), time)
    elif time.ndim == 1:
        time = time[:, None]
    if np.any((time < 0) | (time > 1)):
        raise ValueError("time must be in [0,1]")
    t = time[:, :, None]
    sx = t ** 2 * target_cov + (1 - t) ** 2 * source_cov
    ux = t * target_cov - (1 - t) * source_cov
    centered = x - time * target_mean
    solved = np.linalg.solve(sx, centered[..., None])[..., 0]
    return target_mean + np.einsum("nij,nj->ni", ux, solved)


def conditional_oracle_noise(time, target_cov, source_cov):
    """Conditional irreducible velocity covariance for the privileged oracle."""
    time = np.asarray(time, dtype=np.float64).reshape(-1, 1, 1)
    sx = time ** 2 * target_cov + (1 - time) ** 2 * source_cov
    ux = time * target_cov - (1 - time) * source_cov
    covariance = target_cov + source_cov - ux @ np.linalg.solve(sx, ux.swapaxes(-1, -2))
    return (covariance + covariance.swapaxes(-1, -2)) / 2


def sample_conditional_target(config: GPConfig, dataset: GPDataset, samples=1, seed=0):
    """Exact future draws conditional on raw history, in normalized coordinates.

    Returns [N,S,F]. This is the true GP conditional distribution with privileged
    raw-history statistics and provides a sampling/roughness reference.
    """
    if samples < 1:
        raise ValueError("samples must be positive")
    mean, covariance = conditional_target(config, dataset)
    chol = np.linalg.cholesky(covariance)
    z = _rng(seed).standard_normal((len(dataset), samples, config.horizon))
    return mean[:, None, :] + np.einsum("nsj,nij->nsi", z, chol)
