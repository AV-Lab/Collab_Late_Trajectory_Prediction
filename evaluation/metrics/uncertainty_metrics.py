import numpy as np
from scipy.special import ndtr
from scipy.stats import rankdata


CONFIDENCE_LEVELS = tuple(index / 10.0 for index in range(1, 10))


def chi_square_thresholds(confidence_levels=CONFIDENCE_LEVELS):
    """Two-dimensional chi-square thresholds for Gaussian confidence regions."""
    levels = np.asarray(confidence_levels, dtype=np.float64)
    if levels.ndim != 1 or levels.size == 0 or np.any((levels <= 0) | (levels >= 1)):
        raise ValueError("Confidence levels must lie strictly between zero and one.")
    return -2.0 * np.log1p(-levels)


def squared_mahalanobis_errors(prediction, target, covariance, jitter=1e-6):
    """Squared Mahalanobis errors for aligned 2D Gaussian positions."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("Prediction and target must have matching shape [T, 2].")
    expected_covariance_shape = (prediction.shape[0], 2, 2)
    if covariance.shape != expected_covariance_shape:
        raise ValueError(
            f"Covariance shape differs: {covariance.shape} != {expected_covariance_shape}."
        )
    if prediction.shape[0] == 0:
        raise ValueError("At least one future timestep is required.")
    if jitter <= 0:
        raise ValueError("Covariance jitter must be positive.")
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(target).all()
        and np.isfinite(covariance).all()
    ):
        raise ValueError("Uncertainty inputs must contain only finite values.")

    covariance = 0.5 * (covariance + np.swapaxes(covariance, -1, -2))
    covariance = covariance + float(jitter) * np.eye(2, dtype=np.float64)
    if np.any(np.linalg.eigvalsh(covariance) <= 0):
        raise ValueError("Gaussian covariances must be positive definite.")
    difference = target - prediction
    solved = np.linalg.solve(covariance, difference[..., None]).squeeze(-1)
    return np.einsum("ti,ti->t", difference, solved)


def msne(mahalanobis_squared, dimension=2):
    """Mean normalized squared error from squared Mahalanobis values."""
    values = np.asarray(mahalanobis_squared, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if dimension <= 0:
        raise ValueError("Output dimension must be positive.")
    return float(values.mean() / float(dimension))


def calibration_curve(
    mahalanobis_squared,
    confidence_levels=CONFIDENCE_LEVELS,
):
    """Return expected confidence, empirical coverage, thresholds, and ECE."""
    values = np.asarray(mahalanobis_squared, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    levels = np.asarray(confidence_levels, dtype=np.float64)
    thresholds = chi_square_thresholds(levels)
    if values.size == 0:
        coverage = np.full(levels.shape, np.nan, dtype=np.float64)
        ece_value = float("nan")
    else:
        coverage = (values[:, None] <= thresholds[None, :]).mean(axis=0)
        ece_value = ece(coverage, levels)
    return {
        "confidence": levels.tolist(),
        "coverage": coverage.tolist(),
        "thresholds": thresholds.tolist(),
        "ECE": ece_value,
        "coverage_ECE_L1": ece_value,
    }


def ece(empirical_coverage, confidence_levels=CONFIDENCE_LEVELS):
    """Mean absolute gap between empirical and expected Gaussian coverage."""
    coverage = np.asarray(empirical_coverage, dtype=np.float64).reshape(-1)
    levels = np.asarray(confidence_levels, dtype=np.float64).reshape(-1)
    if coverage.shape != levels.shape:
        raise ValueError("Coverage and confidence levels must have matching shapes.")
    if not np.isfinite(coverage).all():
        return float("nan")
    return float(np.abs(coverage - levels).mean())


def validate_calibration_settings(levels, weights=None):
    """Validate CDF thresholds and weights without rescaling supplied weights.

    Uniform weights summing to one are our default configuration. Explicit
    nonnegative weights may have any positive sum, as in the CE definition.
    """
    levels = np.asarray(levels, dtype=np.float64)
    if (
        levels.ndim != 1
        or levels.size == 0
        or not np.isfinite(levels).all()
        or np.any((levels < 0.0) | (levels > 1.0))
        or np.any(np.diff(levels) <= 0.0)
    ):
        raise ValueError("CDF confidence levels must be increasing finite values in [0, 1].")
    if weights is None:
        weights = np.full(levels.shape, 1.0 / levels.size, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if (
            weights.shape != levels.shape
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
        ):
            raise ValueError("CE weights must align with levels, be finite and nonnegative, and include a positive weight.")
    return levels.copy(), weights.copy()


def _aligned_finite_vectors(*values):
    """Require already aligned samples; never filter individual inputs."""
    arrays = tuple(np.asarray(value, dtype=np.float64) for value in values)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("Sample inputs must be one-dimensional arrays.")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("Sample inputs must have matching shapes.")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("Sample inputs must contain only finite values.")
    return arrays


def gaussian_cdf_calibration(
    mean,
    target,
    variance,
    confidence_levels=CONFIDENCE_LEVELS,
    weights=None,
):
    """Scalar Gaussian predictive-CDF CE using weighted squared gaps.

    ``coverage[j]`` is the fraction of PIT values at or below threshold j.
    Calling this separately on x and y tests marginal calibration, not the
    complete joint distribution. Variance must already include any jitter.
    """
    levels, weights = validate_calibration_settings(confidence_levels, weights)
    mean, target, variance = _aligned_finite_vectors(mean, target, variance)
    if np.any(variance <= 0.0):
        raise ValueError("Gaussian variance must be positive.")
    if mean.size == 0:
        coverage = np.full(levels.shape, np.nan, dtype=np.float64)
        ce_value = float("nan")
        diagnostic = "no_valid_samples"
    else:
        # Extreme finite residuals may saturate the CDF legitimately.
        with np.errstate(over="ignore"):
            pit = ndtr((target - mean) / np.sqrt(variance))
        coverage = (pit[:, None] <= levels[None, :]).mean(axis=0)
        ce_value = float(np.sum(weights * np.square(levels - coverage)))
        diagnostic = None
    return {
        "CE": ce_value,
        "confidence": levels.tolist(),
        "coverage": coverage.tolist(),
        "weights": weights.tolist(),
        "count": int(mean.size),
        "diagnostic": diagnostic,
    }


def gaussian_nll(mean, target, covariance, jitter=0.0):
    """Return Gaussian negative log density for each aligned sample.

    Scalar inputs have shape [N], with variances [N]. Joint inputs have shape
    [N, D], with full covariances [N, D, D]. The joint distribution covers the
    D coordinates of each sample, not dependence between future timesteps.
    Covariance is symmetrized and the supplied nonnegative jitter is added
    once. Negative NLL values are valid for continuous probability densities.
    """
    mean = np.asarray(mean, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    if mean.shape != target.shape or mean.ndim not in (1, 2):
        raise ValueError("Gaussian mean and target must have matching shapes [N] or [N, D].")
    if mean.ndim == 2 and mean.shape[1] == 0:
        raise ValueError("Gaussian output dimension must be positive.")
    if not np.isscalar(jitter) or not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("Covariance jitter must be finite and nonnegative.")
    if mean.ndim == 1:
        if covariance.shape != mean.shape:
            raise ValueError("Scalar Gaussian variances must have shape [N].")
        mean = mean[:, None]
        target = target[:, None]
        covariance = covariance[:, None, None]
    dimension = mean.shape[1]
    if covariance.shape != (len(mean), dimension, dimension):
        raise ValueError("Gaussian covariances must have shape [N, D, D].")
    if not all(np.isfinite(value).all() for value in (mean, target, covariance)):
        raise ValueError("Gaussian inputs must contain only finite values.")
    if len(mean) == 0:
        return np.empty(0, dtype=np.float64)

    covariance = 0.5 * covariance + 0.5 * np.swapaxes(covariance, -1, -2)
    covariance = covariance + float(jitter) * np.eye(dimension, dtype=np.float64)
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Gaussian covariances must be positive definite after jitter.") from exc
    with np.errstate(over="ignore", invalid="ignore"):
        whitened = np.linalg.solve(cholesky, (target - mean)[..., None])[..., 0]
        squared_distance = np.sum(np.square(whitened), axis=-1)
        log_determinant = 2.0 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(axis=-1)
        nll = 0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + squared_distance)
    if not np.isfinite(nll).all():
        raise ValueError("Gaussian NLL is non-finite for the supplied inputs.")
    return nll


def _error_uncertainty_vectors(errors, uncertainties):
    errors, uncertainties = _aligned_finite_vectors(errors, uncertainties)
    if np.any(errors < 0.0) or np.any(uncertainties < 0.0):
        raise ValueError("Prediction errors and variances must be nonnegative.")
    return errors, uncertainties


def sparsification_curve(errors, uncertainties, max_points=201):
    """Normalized uncertainty/oracle curves and their exact finite-sample AUSE.

    Remove the highest uncertainties or, independently, the highest errors.
    The score is the left-Riemann sum on removal fractions k/N, k=0,...,N-1,
    with width 1/N: every retained set is nonempty. Uncertainty ties use the
    expected retained error under uniform random ordering within each tied
    group. Plot arrays alone are downsampled to ``max_points``; the score uses
    every fraction. ``error`` is the normalized difference between curves.

    Scalar callers use absolute error and variance. The explicitly separate
    2D adaptation uses Euclidean error and covariance trace.
    """
    errors, uncertainties = _error_uncertainty_vectors(errors, uncertainties)
    if (
        isinstance(max_points, (bool, np.bool_))
        or not isinstance(max_points, (int, np.integer))
        or max_points < 2
    ):
        raise ValueError("max_points must be an integer of at least two.")
    if errors.size == 0:
        return {
            "AUSE": float("nan"),
            "removal_fraction": [],
            "uncertainty": [],
            "oracle": [],
            "error": [],
            "diagnostic": "no_valid_samples",
        }

    count = len(errors)
    retained_counts = np.arange(1, count + 1, dtype=np.float64)
    maximum_error = float(errors.max())
    if maximum_error == 0.0:
        uncertainty_curve = np.zeros(count, dtype=np.float64)
        oracle_curve = np.zeros(count, dtype=np.float64)
        diagnostic = "zero_full_error"
    else:
        # Scaling before taking the mean also avoids summation overflow.
        normalized_errors = errors / maximum_error
        normalized_errors /= normalized_errors.mean()
        order = np.argsort(uncertainties, kind="stable")
        ordered_uncertainties = uncertainties[order]
        ordered_errors = normalized_errors[order]
        starts = np.r_[0, np.flatnonzero(np.diff(ordered_uncertainties)) + 1]
        lengths = np.diff(np.r_[starts, count])
        group_means = np.add.reduceat(ordered_errors, starts) / lengths
        expected_errors = np.repeat(group_means, lengths)
        uncertainty_curve = (np.cumsum(expected_errors) / retained_counts)[::-1]
        oracle_curve = (np.cumsum(np.sort(normalized_errors)) / retained_counts)[::-1]
        diagnostic = None

    # Oracle error is a lower bound; clip floating-point roundoff at zero.
    difference = np.maximum(uncertainty_curve - oracle_curve, 0.0)
    score = float(difference.mean())
    indices = np.unique(np.linspace(0, count - 1, min(count, int(max_points)), dtype=int))
    removal_fraction = np.arange(count, dtype=np.float64) / count
    return {
        "AUSE": score,
        "removal_fraction": removal_fraction[indices].tolist(),
        "uncertainty": uncertainty_curve[indices].tolist(),
        "oracle": oracle_curve[indices].tolist(),
        "error": difference[indices].tolist(),
        "diagnostic": diagnostic,
    }


def ause(errors, uncertainties):
    """Return AUSE with the protocol documented by sparsification_curve."""
    return sparsification_curve(errors, uncertainties)["AUSE"]


def spearman_error_uncertainty(errors, uncertainties):
    """Signed Pearson correlation of minimum-tie uncertainty/error ranks."""
    errors, uncertainties = _error_uncertainty_vectors(errors, uncertainties)
    if len(errors) < 2:
        return {
            "Spearman": float("nan"),
            "diagnostic": "no_valid_samples" if len(errors) == 0 else "fewer_than_two_samples",
        }
    error_ranks = rankdata(errors, method="min").astype(np.float64)
    uncertainty_ranks = rankdata(uncertainties, method="min").astype(np.float64)
    error_ranks -= error_ranks.mean()
    uncertainty_ranks -= uncertainty_ranks.mean()
    if not np.any(error_ranks) or not np.any(uncertainty_ranks):
        return {"Spearman": float("nan"), "diagnostic": "constant_ranks"}
    coefficient = np.dot(error_ranks, uncertainty_ranks) / np.sqrt(
        np.dot(error_ranks, error_ranks) * np.dot(uncertainty_ranks, uncertainty_ranks)
    )
    return {"Spearman": float(np.clip(coefficient, -1.0, 1.0)), "diagnostic": None}
