"""Train-fitted PCA and preregistered evaluation in observation coordinates."""
from dataclasses import dataclass
import numpy as np


@dataclass
class PCA:
    mean: np.ndarray
    components: np.ndarray  # rows, descending explained variance
    eigenvalues: np.ndarray
    floor: float

    @property
    def scales(self):
        return np.sqrt(np.maximum(self.eigenvalues, self.floor))

    def project(self, values, center=True):
        values = np.asarray(values)
        return (values - self.mean if center else values) @ self.components.T

    def whiten(self, values, center=True):
        return self.project(values, center=center) / self.scales

    def unwhiten(self, values, center=True):
        raw = (np.asarray(values) * self.scales) @ self.components
        return raw + self.mean if center else raw

    def to_dict(self):
        return {"mean": self.mean.tolist(), "components": self.components.tolist(),
                "eigenvalues": self.eigenvalues.tolist(), "floor": float(self.floor)}


def fit_pca(training_values, relative_floor=1e-6, absolute_floor=1e-8):
    """Fit ONLY on training velocities/targets, using unbiased sample covariance.

    The floor affects whitening/normalized diagnostics, never the spectrum or raw
    projected errors. Relative floor is relative to the leading eigenvalue.
    """
    x = np.asarray(training_values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("PCA requires finite [N,F] data and N >= 2")
    mean = x.mean(axis=0)
    cov = (x - mean).T @ (x - mean) / (len(x) - 1)
    eigenvalues, vectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0)
    vectors = vectors[:, order].T
    # Fix arbitrary eigenvector signs for reproducible figures/checkpoints.
    pivot = np.argmax(np.abs(vectors), axis=1)
    signs = np.sign(vectors[np.arange(len(vectors)), pivot])
    vectors *= np.where(signs == 0, 1, signs)[:, None]
    floor = max(float(eigenvalues[0]) * relative_floor, absolute_floor)
    return PCA(mean, vectors, eigenvalues, floor)


def spectrum(values, pca: PCA):
    """Held-out variance captured by TRAIN axes, centered at held-out mean.

    Re-centering only estimates held-out variance and is not refitting directions.
    """
    x = np.asarray(values, dtype=np.float64)
    pc_variance = pca.project(x, center=False).var(axis=0, ddof=1)
    total = float(pc_variance.sum())
    energy = pc_variance / total if total > 0 else np.zeros_like(pc_variance)
    entropy_rank = float(np.exp(-np.sum(energy[energy > 0] * np.log(energy[energy > 0]))))
    return {"pc_variance": pc_variance.tolist(), "energy": energy.tolist(),
            "top1_energy": float(energy[0]), "top2_energy": float(energy[:2].sum()),
            "total_variance": total, "effective_rank": entropy_rank,
            "roughness": roughness(x), "mean_within_patch_std": float(x.std(axis=1).mean())}


def roughness(paths):
    """Mean |adjacent jump|; average equally over all paths and F-1 differences."""
    paths = np.asarray(paths, dtype=np.float64)
    if paths.shape[-1] < 2 or not np.isfinite(paths).all():
        raise ValueError("Finite paths of length >=2 are required")
    return float(np.mean(np.abs(np.diff(paths, axis=-1))))


def velocity_metrics(prediction, target_velocity, pca: PCA):
    """MSE of error projected on each velocity PC (errors are NOT centered).

    Raw residual MSE is the mean over individual PCs3..F, so it has the same
    units and coordinate count normalization as PC1 MSE. The secondary scaled
    scores divide each PC error by its train variance (regularized by pca.floor).
    """
    prediction, target_velocity = np.asarray(prediction), np.asarray(target_velocity)
    if prediction.shape != target_velocity.shape or prediction.ndim != 2 or prediction.shape[1] < 3:
        raise ValueError("prediction/velocity must be matching [N,F] arrays, F>=3")
    if not np.isfinite(prediction).all() or not np.isfinite(target_velocity).all():
        raise ValueError("velocity arrays must be finite")
    errors = pca.project(prediction - target_velocity, center=False)
    pc_mse = np.mean(errors ** 2, axis=0)
    normalized = pc_mse / np.maximum(pca.eigenvalues, pca.floor)
    return {"velocity_mse": float(np.mean((prediction - target_velocity) ** 2)),
            "pc1_mse": float(pc_mse[0]), "pc2_mse": float(pc_mse[1]),
            "residual_mse": float(pc_mse[2:].mean()), "pc_mse": pc_mse.tolist(),
            "pc1_normalized_mse": float(normalized[0]),
            "pc2_normalized_mse": float(normalized[1]),
            "residual_normalized_mse": float(normalized[2:].mean()),
            "pc_normalized_mse": normalized.tolist(),
            "variance_floor": float(pca.floor)}


def success_rule(pc1_error, residual_error, generated_roughness, target_roughness, tolerance=0.20):
    """Literal preregistered TWO-SIDED 'within 20%' criterion from assignment.

    A residual error smaller than .8*PC1 does not pass this literal criterion.
    This odd consequence is exposed, not silently changed after seeing results.
    """
    values = np.array([pc1_error, residual_error, generated_roughness, target_roughness], float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Errors and roughness must be finite nonnegative numbers")
    error_ratio = None if pc1_error == 0 else float(residual_error / pc1_error)
    roughness_ratio = None if target_roughness == 0 else float(generated_roughness / target_roughness)
    error_pass = bool(pc1_error > 0 and (1 - tolerance) * pc1_error <= residual_error <= (1 + tolerance) * pc1_error)
    roughness_pass = bool(target_roughness > 0 and (1 - tolerance) * target_roughness <= generated_roughness <= (1 + tolerance) * target_roughness)
    return {"residual_to_pc1": error_ratio, "roughness_ratio": roughness_ratio,
            "velocity_rule_pass": error_pass, "roughness_rule_pass": roughness_pass,
            "works": error_pass and roughness_pass}


def evaluate(prediction, target_velocity, generated_paths, target_paths, pca: PCA):
    metrics = velocity_metrics(prediction, target_velocity, pca)
    metrics["generated_roughness"] = roughness(generated_paths)
    metrics["target_roughness"] = roughness(target_paths)
    metrics["success"] = success_rule(metrics["pc1_mse"], metrics["residual_mse"],
                                      metrics["generated_roughness"], metrics["target_roughness"])
    metrics["normalized_success_diagnostic"] = success_rule(
        metrics["pc1_normalized_mse"], metrics["residual_normalized_mse"],
        metrics["generated_roughness"], metrics["target_roughness"])
    return metrics


def oracle_noise_metrics(covariances, pca: PCA):
    covariance = np.asarray(covariances).mean(axis=0)
    pc_noise = np.diag(pca.components @ covariance @ pca.components.T)
    return {"pc1_noise": float(pc_noise[0]), "pc2_noise": float(pc_noise[1]),
            "residual_noise": float(pc_noise[2:].mean()), "pc_noise": pc_noise.tolist()}


def paired_bootstrap_difference(a, b, seed=0, repetitions=2000, confidence=0.95):
    """Paired mean difference CI; observations must be independent seeds/windows.

    Prefer seeds for primary model comparisons. Do not treat F correlated time
    points or multiple generations from one context as independent observations.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("Matching 1D arrays with at least two pairs are required")
    diff = a - b
    draws = np.random.default_rng(seed).integers(0, len(a), size=(repetitions, len(a)))
    boot = diff[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    return {"mean_difference": float(diff.mean()), "ci_low": float(np.quantile(boot, alpha)),
            "ci_high": float(np.quantile(boot, 1 - alpha)), "pairs": len(a)}
