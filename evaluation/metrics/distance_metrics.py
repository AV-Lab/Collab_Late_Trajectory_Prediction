import numpy as np


HIT_THRESHOLDS = (0.5, 1.0)


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _validate_inputs(prediction, target, mask=None):
    prediction = _to_numpy(prediction)
    target = _to_numpy(target)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {prediction.shape} != {target.shape}."
        )
    if prediction.ndim not in (2, 3) or prediction.shape[-1] != 2:
        raise ValueError("Prediction and target must have shape [T, 2] or [N, T, 2].")
    if prediction.shape[-2] == 0:
        raise ValueError("At least one future timestep is required.")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Prediction and target must contain only finite values.")

    expected_mask_shape = prediction.shape[:-1]
    if mask is None:
        mask = np.ones(expected_mask_shape, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != expected_mask_shape:
            raise ValueError(
                f"Mask shape differs: {mask.shape} != {expected_mask_shape}."
            )
    return prediction, target, mask


def ade(prediction, target, mask=None):
    """Mean per-trajectory displacement error over valid future points."""
    prediction, target, mask = _validate_inputs(prediction, target, mask)
    errors = np.linalg.norm(prediction - target, axis=-1)
    if errors.ndim == 1:
        if not mask.any():
            return float("nan")
        return float(errors[mask].mean())

    valid = mask.any(axis=-1)
    if not valid.any():
        return float("nan")
    lengths = mask.sum(axis=-1).clip(min=1)
    per_trajectory = (errors * mask).sum(axis=-1) / lengths
    return float(per_trajectory[valid].mean())


def mse_2d(prediction, target, mask=None):
    """Mean squared position error (dx² + dy²), in square metres.

    Average valid points within each forecast, then weight nonempty forecasts
    equally, as in ADE. This is not a mean over the two coordinates.
    """
    prediction, target, mask = _validate_inputs(prediction, target, mask)
    squared_errors = np.square(prediction - target).sum(axis=-1)
    if squared_errors.ndim == 1:
        if not mask.any():
            return float("nan")
        return float(squared_errors[mask].mean())

    valid = mask.any(axis=-1)
    if not valid.any():
        return float("nan")
    lengths = mask.sum(axis=-1).clip(min=1)
    per_trajectory = (squared_errors * mask).sum(axis=-1) / lengths
    return float(per_trajectory[valid].mean())


def fde(prediction, target, mask=None):
    """Mean displacement error at each trajectory's final valid timestep."""
    prediction, target, mask = _validate_inputs(prediction, target, mask)
    errors = np.linalg.norm(prediction - target, axis=-1)
    if errors.ndim == 1:
        valid_indices = np.flatnonzero(mask)
        return float(errors[valid_indices[-1]]) if valid_indices.size else float("nan")

    valid = mask.any(axis=-1)
    if not valid.any():
        return float("nan")
    time_indices = np.arange(mask.shape[-1])
    final_indices = np.where(mask, time_indices, 0).max(axis=-1)
    rows = np.arange(errors.shape[0])
    return float(errors[rows[valid], final_indices[valid]].mean())


def hit_rate(final_errors, threshold):
    """Fraction of finite final errors at or below a metre threshold."""
    threshold = float(threshold)
    if threshold <= 0:
        raise ValueError("Hit-rate threshold must be positive.")
    values = _to_numpy(final_errors).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.mean(values <= threshold))
