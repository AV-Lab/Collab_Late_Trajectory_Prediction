"""Shared, validated uncertainty samples and point-weighted aggregation.

Prepare each forecast once, then share its immutable batch between the overall,
scenario, class, and source accumulators. Covariances are symmetrized and receive
jitter only here; every downstream metric uses the same surviving sample rows.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.special import ndtr

from evaluation.metrics.uncertainty_metrics import (
    CONFIDENCE_LEVELS,
    gaussian_cdf_calibration,
    gaussian_nll,
    sparsification_curve,
    spearman_error_uncertainty,
    validate_calibration_settings,
)


def _readonly(values, dtype=None):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _freeze_identifier(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_identifier(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_identifier(item) for item in value)
    return value


def _plain_identifier(value):
    if isinstance(value, Mapping):
        return {key: _plain_identifier(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_identifier(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class UncertaintyBatch:
    """An immutable forecast batch; feature arrays contain valid rows only.

    ``valid_mask``, ``exclusion_reasons``, and ``mahalanobis_squared`` retain the
    original row count. ``valid_indices`` maps feature rows back to that input.
    A shared identifier mapping describes a forecast; exported ``horizon_step``
    is its original one-based row index, including holes left by exclusions.
    """

    candidate_count: int
    valid_mask: np.ndarray
    valid_indices: np.ndarray
    exclusion_reasons: tuple
    mahalanobis_squared: np.ndarray
    means: np.ndarray
    targets: np.ndarray
    covariances: np.ndarray
    absolute_errors: np.ndarray
    variances: np.ndarray
    pit: np.ndarray
    position_errors: np.ndarray
    covariance_trace: np.ndarray
    joint_nll: np.ndarray
    marginal_nll: np.ndarray
    identifiers: object
    timestamps: object
    jitter: float

    @property
    def valid_count(self):
        return int(self.valid_indices.size)

    @property
    def excluded_count(self):
        return self.candidate_count - self.valid_count

    @property
    def exclusion_counts(self):
        return dict(Counter(reason for reason in self.exclusion_reasons if reason is not None))

    def identifier_at(self, index):
        context = self.identifiers if isinstance(self.identifiers, Mapping) else self.identifiers[index]
        result = _plain_identifier(context)
        result.setdefault("horizon_step", int(index) + 1)
        if self.timestamps is not None:
            result.setdefault("horizon_seconds", float(self.timestamps[index]))
        return result


def prepare_uncertainty_batch(means, target, covariances, jitter=1e-6, identifiers=None, timestamps=None):
    """Validate aligned ``[N,2]`` positions and construct one shared sample mask.

    ``None`` covariance, or an entirely NaN 2x2 row, denotes missing covariance.
    Other nonfinite covariance entries are classified separately. Exclusion
    reasons are mutually exclusive, in the order checked below. Zero jitter is
    permitted for already regularized inputs; otherwise jitter is added once.
    """
    means = np.asarray(means, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 2 or means.shape != target.shape:
        raise ValueError("Means and target must have matching shape [N, 2].")
    jitter = float(jitter)
    if not np.isfinite(jitter) or jitter < 0:
        raise ValueError("Covariance jitter must be finite and nonnegative.")
    count = len(means)
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.shape != (count,) or not np.isfinite(timestamps).all():
            raise ValueError("Timestamps must be N finite relative forecast times.")
        timestamps = _readonly(timestamps)
    if covariances is None:
        covariances = np.full((count, 2, 2), np.nan)
    else:
        covariances = np.asarray(covariances, dtype=np.float64)
        if covariances.shape != (count, 2, 2):
            raise ValueError("Covariances must have shape [N, 2, 2].")
    if identifiers is None:
        identifiers = MappingProxyType({})
    elif isinstance(identifiers, Mapping):
        identifiers = _freeze_identifier(identifiers)
    else:
        identifiers = tuple(identifiers)
        if len(identifiers) != count or any(not isinstance(item, Mapping) for item in identifiers):
            raise ValueError("Identifiers must be a forecast mapping or N point mappings.")
        identifiers = tuple(_freeze_identifier(item) for item in identifiers)

    reasons = np.full(count, None, dtype=object)

    def exclude(mask, reason):
        reasons[(reasons == None) & mask] = reason  # noqa: E711

    exclude(~np.isfinite(means).all(axis=1), "nonfinite_prediction")
    exclude(~np.isfinite(target).all(axis=1), "nonfinite_target")
    exclude(np.isnan(covariances).all(axis=(1, 2)), "missing_covariance")
    exclude(~np.isfinite(covariances).all(axis=(1, 2)), "nonfinite_covariance")

    # Multiplying before addition avoids overflow when both entries are large.
    with np.errstate(over="ignore", invalid="ignore"):
        regularized = 0.5 * covariances + 0.5 * covariances.swapaxes(-1, -2)
        regularized = regularized + jitter * np.eye(2)
    exclude(~np.isfinite(regularized).all(axis=(1, 2)), "nonfinite_regularized_covariance")
    indices = np.flatnonzero(reasons == None)  # noqa: E711
    if indices.size:
        try:
            eigenvalues = np.linalg.eigvalsh(regularized[indices])
            positive = np.isfinite(eigenvalues).all(axis=1) & (eigenvalues > 0).all(axis=1)
        except np.linalg.LinAlgError:
            positive = np.zeros(indices.size, dtype=bool)
            for offset, index in enumerate(indices):
                try:
                    values = np.linalg.eigvalsh(regularized[index])
                    positive[offset] = np.isfinite(values).all() and (values > 0).all()
                except np.linalg.LinAlgError:
                    pass
        reasons[indices[~positive]] = "not_positive_definite"

    indices = np.flatnonzero(reasons == None)  # noqa: E711
    difference = np.full((count, 2), np.nan)
    absolute_errors = np.full((count, 2), np.nan)
    variances = np.diagonal(regularized, axis1=-2, axis2=-1).copy()
    pit = np.full((count, 2), np.nan)
    position_errors = np.full(count, np.nan)
    trace = np.full(count, np.nan)
    d_squared = np.full(count, np.nan)
    joint_nll = np.full(count, np.nan)
    marginal_nll = np.full((count, 2), np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        difference[indices] = target[indices] - means[indices]
        absolute_errors[indices] = np.abs(difference[indices])
        position_errors[indices] = np.hypot(difference[indices, 0], difference[indices, 1])
        trace[indices] = variances[indices].sum(axis=1)
        pit[indices] = ndtr(difference[indices] / np.sqrt(variances[indices]))
        if indices.size:
            # Whitening avoids forming an inverse and retains cross-covariance.
            try:
                cholesky = np.linalg.cholesky(regularized[indices])
                whitened = np.linalg.solve(cholesky, difference[indices, :, None])[..., 0]
                d_squared[indices] = np.sum(whitened * whitened, axis=1)
            except np.linalg.LinAlgError:
                for index in indices:
                    try:
                        chol = np.linalg.cholesky(regularized[index])
                        whitened = np.linalg.solve(chol, difference[index])
                        d_squared[index] = np.dot(whitened, whitened)
                    except np.linalg.LinAlgError:
                        reasons[index] = "not_positive_definite"

    derived_finite = (
        np.isfinite(difference).all(axis=1)
        & np.isfinite(absolute_errors).all(axis=1)
        & np.isfinite(variances).all(axis=1)
        & np.isfinite(pit).all(axis=1)
        & np.isfinite(position_errors)
        & np.isfinite(trace)
        & np.isfinite(d_squared)
    )
    exclude(~derived_finite, "nonfinite_derived_features")
    indices = np.flatnonzero(reasons == None)  # noqa: E711
    if indices.size:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
            try:
                joint_nll[indices] = gaussian_nll(means[indices], target[indices], regularized[indices], jitter=0.0)
                for coordinate in range(2):
                    marginal_nll[indices, coordinate] = gaussian_nll(
                        means[indices, coordinate], target[indices, coordinate], variances[indices, coordinate], jitter=0.0,
                    )
            except (ValueError, np.linalg.LinAlgError):
                # A numerical failure at one point must not discard its peers.
                for index in indices:
                    try:
                        joint_nll[index] = gaussian_nll(means[index:index + 1], target[index:index + 1], regularized[index:index + 1], jitter=0.0)[0]
                        for coordinate in range(2):
                            marginal_nll[index, coordinate] = gaussian_nll(
                                means[index:index + 1, coordinate], target[index:index + 1, coordinate], variances[index:index + 1, coordinate], jitter=0.0,
                            )[0]
                    except (ValueError, np.linalg.LinAlgError):
                        reasons[index] = "nonfinite_derived_features"
    exclude(~np.isfinite(joint_nll) | ~np.isfinite(marginal_nll).all(axis=1), "nonfinite_derived_features")
    valid_mask = reasons == None  # noqa: E711
    indices = np.flatnonzero(valid_mask)
    d_squared[~valid_mask] = np.nan
    return UncertaintyBatch(
        candidate_count=count,
        valid_mask=_readonly(valid_mask),
        valid_indices=_readonly(indices),
        exclusion_reasons=tuple(reasons),
        mahalanobis_squared=_readonly(d_squared),
        means=_readonly(means[indices]),
        targets=_readonly(target[indices]),
        covariances=_readonly(regularized[indices]),
        absolute_errors=_readonly(absolute_errors[indices]),
        variances=_readonly(variances[indices]),
        pit=_readonly(pit[indices]),
        position_errors=_readonly(position_errors[indices]),
        covariance_trace=_readonly(trace[indices]),
        joint_nll=_readonly(joint_nll[indices]),
        marginal_nll=_readonly(marginal_nll[indices]),
        identifiers=identifiers,
        timestamps=timestamps,
        jitter=jitter,
    )


class UncertaintyAccumulator:
    """Pool valid points globally; no averaging of per-forecast CE or AUSE."""

    def __init__(self, confidence_levels=CONFIDENCE_LEVELS, ce_weights=None):
        levels, weights = validate_calibration_settings(confidence_levels, ce_weights)
        self.confidence_levels = _readonly(levels)
        self.ce_weights = _readonly(weights)
        self._explicit_ce_weights = ce_weights is not None
        self._batches = []

    def add_batch(self, batch):
        if not isinstance(batch, UncertaintyBatch):
            raise TypeError("add_batch requires a prepared UncertaintyBatch.")
        self._batches.append(batch)

    def _concatenate(self, name, trailing_shape=()):
        arrays = [getattr(batch, name) for batch in self._batches if batch.valid_count]
        if not arrays:
            return np.empty((0,) + trailing_shape, dtype=np.float64)
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)

    def metadata(self):
        return {
            "sample_unit": "matched forecast timestep",
            "validity": "shared finite means/target/covariance, positive definite symmetrized covariance, finite derived features",
            "covariance_regularization": "symmetrize, then add jitter once before all uncertainty metrics",
            "covariance_jitter_values": sorted({batch.jitter for batch in self._batches}),
            "confidence_levels": self.confidence_levels.tolist(),
            "ce_weights": self.ce_weights.tolist(),
            "ce_weight_policy": "explicit weights, not renormalized" if self._explicit_ce_weights else "uniform 1 / number of confidence levels",
            "ce_definition": "weighted squared marginal Gaussian CDF calibration error",
            "uncertainty_x_y": "marginal variance",
            "uncertainty_2D": "covariance trace",
            "error_x_y": "absolute coordinate error",
            "error_2D": "Euclidean position error",
            "aggregation": "metrics computed over concatenated valid points within each reporting group",
            "ause_normalization": "each retained mean error divided by full-sample mean error",
            "ause_integration": "left Riemann sum at k/N for k=0,...,N-1, each interval width 1/N; no empty retained set",
            "ause_uncertainty_ties": "expected retained error under uniform ordering within tied uncertainty groups",
            "ause_zero_error": "AUSE=0 and zero curves when every error is zero",
            "spearman_definition": "signed Pearson correlation of error and uncertainty ranks",
            "spearman_ties": "minimum rank within each tie group",
            "spearman_degenerate": "NaN plus diagnostic for fewer than two samples or constant ranks",
            "curve_max_points": 201,
            "nll_2D": "joint Gaussian NLL using the full 2x2 covariance",
            "nll_reductions": "NLL_sum sums valid pointwise joint XY NLL; NLL_mean divides that sum by valid point count; marginal x/y reductions reported separately",
            "nll_temporal_scope": "pointwise joint XY distribution; no temporal joint covariance is modeled",
        }

    def compute(self):
        means = self._concatenate("means", (2,))
        targets = self._concatenate("targets", (2,))
        variances = self._concatenate("variances", (2,))
        errors = self._concatenate("absolute_errors", (2,))
        position_errors = self._concatenate("position_errors")
        traces = self._concatenate("covariance_trace")
        joint_nll = self._concatenate("joint_nll")
        marginal_nll = self._concatenate("marginal_nll", (2,))
        count = len(joint_nll)
        candidate_count = sum(batch.candidate_count for batch in self._batches)
        exclusions = Counter()
        for batch in self._batches:
            exclusions.update(batch.exclusion_counts)
        cdf = {}
        curves = {}
        correlations = {}
        diagnostics = {}
        for coordinate, name in enumerate(("x", "y")):
            cdf[name] = gaussian_cdf_calibration(
                means[:, coordinate], targets[:, coordinate], variances[:, coordinate],
                confidence_levels=self.confidence_levels, weights=self.ce_weights,
            )
        for name, error, uncertainty in (
            ("x", errors[:, 0], variances[:, 0]),
            ("y", errors[:, 1], variances[:, 1]),
            ("position", position_errors, traces),
        ):
            curves[name] = sparsification_curve(error, uncertainty, max_points=201)
            correlations[name] = spearman_error_uncertainty(error, uncertainty)
            diagnostics[name] = {
                "sparsification": curves[name].get("diagnostic"),
                "spearman": correlations[name].get("diagnostic"),
            }
            if name in cdf:
                diagnostics[name]["calibration"] = cdf[name].get("diagnostic")
        ce_x, ce_y = cdf["x"]["CE"], cdf["y"]["CE"]
        return {
            "CE_x": ce_x,
            "CE_y": ce_y,
            "CE_mean": float((ce_x + ce_y) / 2.0),
            "AUSE_x": curves["x"]["AUSE"],
            "AUSE_y": curves["y"]["AUSE"],
            "AUSE_2D": curves["position"]["AUSE"],
            "Spearman_x": correlations["x"]["Spearman"],
            "Spearman_y": correlations["y"]["Spearman"],
            "Spearman_2D": correlations["position"]["Spearman"],
            "NLL_sum": float(joint_nll.sum()),
            "NLL_mean": float(joint_nll.mean()) if count else float("nan"),
            "NLL_x_sum": float(marginal_nll[:, 0].sum()),
            "NLL_y_sum": float(marginal_nll[:, 1].sum()),
            "NLL_x_mean": float(marginal_nll[:, 0].mean()) if count else float("nan"),
            "NLL_y_mean": float(marginal_nll[:, 1].mean()) if count else float("nan"),
            "uncertainty_candidate_count": int(candidate_count),
            "uncertainty_valid_count": int(count),
            "uncertainty_excluded_count": int(candidate_count - count),
            "uncertainty_exclusion_counts": dict(sorted(exclusions.items())),
            "uncertainty_diagnostics": diagnostics,
            "cdf_calibration": cdf,
            "sparsification": curves,
        }

    def sample_records(self, include_excluded=False):
        """Yield opt-in sample exports without materializing them in reports."""
        for batch in self._batches:
            valid_row = 0
            for index in range(batch.candidate_count):
                valid = bool(batch.valid_mask[index])
                if not valid and not include_excluded:
                    continue
                record = {
                    "identifiers": batch.identifier_at(index),
                    "valid": valid,
                    "exclusion_reason": batch.exclusion_reasons[index],
                }
                if valid:
                    record.update({
                        "mean": batch.means[valid_row].tolist(),
                        "target": batch.targets[valid_row].tolist(),
                        "covariance": batch.covariances[valid_row].tolist(),
                        "absolute_error": batch.absolute_errors[valid_row].tolist(),
                        "variance": batch.variances[valid_row].tolist(),
                        "pit": batch.pit[valid_row].tolist(),
                        "position_error": float(batch.position_errors[valid_row]),
                        "covariance_trace": float(batch.covariance_trace[valid_row]),
                        "mahalanobis_squared": float(batch.mahalanobis_squared[index]),
                        "NLL": float(batch.joint_nll[valid_row]),
                        "marginal_NLL": batch.marginal_nll[valid_row].tolist(),
                    })
                    valid_row += 1
                yield record
