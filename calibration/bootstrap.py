"""Estimate raw error second moments with trajectory bootstrap uncertainty.

Fit-only covariance-trace bins or fixed predicted-motion tags group each category
and forecast horizon. Whole trajectories are resampled together so every group
and horizon shares the same draw within its category. The
spectral-norm radius has approximate per-group confidence; it does not establish
conditional coverage, simultaneous confidence, or transfer to deployment data.

All estimation and validation use CPU NumPy and do not read or write files.
"""

import numpy as np

from .motion import MOTION_TAGS


BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_BATCH_SIZE = 50
MIN_VALID_ROWS = 100
SEED = 42
CONFIDENCE = 0.95


# Shared covariance symmetry tolerances for fitting and runtime lookup.
SYMMETRY_ATOL = 1e-12
SYMMETRY_RTOL = 1e-10

EXCLUSION_REASONS = (
    "nonfinite_mean",
    "nonfinite_target",
    "nonfinite_covariance",
    "asymmetric_covariance",
    "non_positive_definite_covariance",
    "nonfinite_normalized_error",
)


def _validate_dataset(means, targets, covariances, categories, sample_fps):
    means = np.asarray(means, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    categories = np.asarray(categories, dtype=object)
    if means.ndim != 3 or means.shape[-1] != 2:
        raise ValueError("means must have shape (N, T, 2).")
    if means.shape[0] == 0 or means.shape[1] == 0:
        raise ValueError("At least one trajectory and future step are required.")
    if targets.shape != means.shape:
        raise ValueError("targets must have the same shape as means.")
    if covariances.shape != means.shape + (2,):
        raise ValueError("covariances must have shape (N, T, 2, 2).")
    if categories.ndim != 1 or len(categories) != len(means):
        raise ValueError("categories must contain one category per trajectory.")
    if any(not isinstance(category, str) or not category.strip() for category in categories):
        raise ValueError("Every category must be a nonempty string.")
    if isinstance(sample_fps, (bool, np.bool_)):
        raise ValueError("sample_fps must be finite and positive.")
    sample_fps = float(sample_fps)
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive.")
    return means, targets, covariances, categories, sample_fps


def _whiten_point(mean, target, covariance):
    """Return (error, normalized error, symmetrized, exclusion reason)."""
    for value, reason in (
        (mean, "nonfinite_mean"),
        (target, "nonfinite_target"),
        (covariance, "nonfinite_covariance"),
    ):
        if not np.isfinite(value).all():
            return None, None, False, reason
    tolerance = SYMMETRY_ATOL + SYMMETRY_RTOL * np.abs(covariance).max()
    with np.errstate(over="ignore", invalid="ignore"):
        asymmetry = np.abs(covariance - covariance.T).max()
    if asymmetry > tolerance:
        return None, None, False, "asymmetric_covariance"
    symmetrized = not np.array_equal(covariance, covariance.T)
    covariance = 0.5 * covariance + 0.5 * covariance.T
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        return None, None, False, "non_positive_definite_covariance"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        error = mean - target
        try:
            normalized = np.linalg.solve(cholesky, error)
        except np.linalg.LinAlgError:
            return None, None, False, "nonfinite_normalized_error"
    if not np.isfinite(error).all() or not np.isfinite(normalized).all():
        return None, None, False, "nonfinite_normalized_error"
    return error, normalized, symmetrized, None


def _valid_category_points(means, targets, covariances):
    """Retain the production estimator's exclusion policy without centering errors."""
    errors = np.zeros_like(means)
    valid = np.zeros(means.shape[:2], dtype=bool)
    traces = np.zeros(means.shape[:2])
    exclusions = dict.fromkeys(EXCLUSION_REASONS, 0)
    for row in range(len(means)):
        for step in range(means.shape[1]):
            error, _, _, reason = _whiten_point(means[row, step], targets[row, step],
                                               covariances[row, step])
            if reason is not None:
                exclusions[reason] += 1
                continue
            errors[row, step] = error
            traces[row, step] = np.trace(covariances[row, step])
            valid[row, step] = True
    return errors, traces, valid, exclusions


def _category_design(category, errors, traces, valid, sample_fps, min_rows):
    """Encode group counts and moment sums so each resample keeps whole forecasts."""
    profiles, features = [], []
    for step in range(errors.shape[1]):
        point_traces = traces[valid[:, step], step]
        edges = (np.unique(np.quantile(point_traces, [.2, .4, .6, .8]))
                 if len(point_traces) else np.array([]))
        assignments = np.searchsorted(edges, traces[:, step], side="right")
        bins = []
        for bin_index in range(len(edges) + 1):
            mask = valid[:, step] & (assignments == bin_index)
            count = int(mask.sum())
            usable = count >= min_rows
            error = errors[mask, step]
            moment = error.T @ error / count if usable else None
            bins.append({
                "uncertainty_bin": bin_index,
                "n_valid": count,
                "usable_for_gate": usable,
                "M_hat": moment.tolist() if usable else None,
                "radius": None,
                "status": "pending" if usable else "insufficient_fit_samples",
            })
            if usable:
                feature = np.zeros((len(errors), 4))
                feature[mask] = np.column_stack((np.ones(count), error[:, 0] ** 2,
                                                 error[:, 0] * error[:, 1], error[:, 1] ** 2))
                features.append(feature)
        profiles.append({
            "category": category,
            "horizon_step": step + 1,
            "horizon_seconds": (step + 1) / sample_fps,
            "uncertainty_cutpoints": edges.tolist(),
            "bins": bins,
        })
    design = np.concatenate(features, axis=1) if features else None
    return profiles, design


def _bootstrap_items(usable, design, rng, draws, confidence):
    """Resample all group/horizon sums using the same whole-trajectory draw."""
    if not usable:
        return
    moments = np.array([item["M_hat"] for item in usable])
    deviations = np.empty((draws, len(usable)))
    count = len(design)
    probabilities = np.full(count, 1.0 / count)
    for start in range(0, draws, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, draws)
        weights = rng.multinomial(count, probabilities, size=stop - start).astype(float)
        totals = (weights @ design).reshape(stop - start, len(usable), 4)
        if np.any(totals[..., 0] == 0):
            raise ValueError("A bootstrap resample has an empty usable group; no certificate was saved.")
        values = totals[..., 1:] / totals[..., :1]
        matrices = np.empty(values.shape[:-1] + (2, 2))
        matrices[..., 0, 0], matrices[..., 0, 1] = values[..., 0], values[..., 1]
        matrices[..., 1, 0], matrices[..., 1, 1] = values[..., 1], values[..., 2]
        deviations[start:stop] = np.abs(np.linalg.eigvalsh(matrices - moments)).max(axis=-1)
    for index, item in enumerate(usable):
        item["radius"] = float(np.quantile(deviations[:, index], confidence))
        item["status"] = "approximate_group_moment"


def _bootstrap_category(profiles, design, rng, draws, confidence):
    usable = [item for row in profiles for item in row["bins"] if item["usable_for_gate"]]
    _bootstrap_items(usable, design, rng, draws, confidence)


def _validate_bootstrap_settings(draws, seed, min_rows, confidence):
    for name, value in (("draws", draws), ("seed", seed), ("min_rows", min_rows)):
        minimum = 0 if name == "seed" else 1
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if isinstance(confidence, bool) or not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")


def estimate_bootstrap_profiles(means, targets, covariances, categories, sample_fps,
                                *, draws=BOOTSTRAP_DRAWS, seed=SEED,
                                min_rows=MIN_VALID_ROWS, confidence=CONFIDENCE):
    """Fit using only the supplied calibration rows, preserving temporal dependence."""
    means, targets, covariances, categories, sample_fps = _validate_dataset(
        means, targets, covariances, categories, sample_fps)
    _validate_bootstrap_settings(draws, seed, min_rows, confidence)
    rng = np.random.default_rng(seed)
    profiles, exclusions = [], {}
    for category in sorted(set(categories)):
        mask = categories == category
        errors, traces, valid, exclusions[category] = _valid_category_points(
            means[mask], targets[mask], covariances[mask])
        category_profiles, design = _category_design(category, errors, traces, valid,
                                                     sample_fps, min_rows)
        _bootstrap_category(category_profiles, design, rng, draws, confidence)
        profiles.extend(category_profiles)
        print(f"Bootstrap fitted: {category}, {int(mask.sum())} trajectories", flush=True)
    return profiles, exclusions


def estimate_motion_profiles(means, targets, covariances, categories, motion_tags, sample_fps,
                             *, draws=BOOTSTRAP_DRAWS, seed=SEED,
                             min_rows=MIN_VALID_ROWS, confidence=CONFIDENCE):
    """Fit direct expected-error risk bounds per category, motion and native step."""
    means, targets, covariances, categories, sample_fps = _validate_dataset(
        means, targets, covariances, categories, sample_fps)
    _validate_bootstrap_settings(draws, seed, min_rows, confidence)
    motion_tags = np.asarray(motion_tags, dtype=object)
    if motion_tags.shape != (len(means),) or any(
            tag is not None and tag not in MOTION_TAGS for tag in motion_tags):
        raise ValueError("motion_tags must contain one known tag or None per trajectory.")
    rng = np.random.default_rng(seed)
    profiles, exclusions = [], {}
    for category in sorted(set(categories)):
        selected = categories == category
        tags = motion_tags[selected]
        errors, _, valid, exclusions[category] = _valid_category_points(
            means[selected], targets[selected], covariances[selected])
        exclusions[category]["unavailable_motion_trajectories"] = int(sum(tag is None for tag in tags))
        category_profiles, features = [], []
        for tag in MOTION_TAGS:
            for step in range(means.shape[1]):
                mask = valid[:, step] & (tags == tag)
                count = int(mask.sum())
                usable = count >= min_rows
                error = errors[mask, step]
                moment = error.T @ error / count if usable else None
                category_profiles.append({
                    "category": category, "motion_tag": tag, "horizon_step": step + 1,
                    "horizon_seconds": (step + 1) / sample_fps, "n_valid": count,
                    "usable_for_gate": usable, "M_hat": moment.tolist() if usable else None,
                    "radius": None, "lower": None, "upper": None,
                    "status": "pending" if usable else "insufficient_fit_samples",
                })
                if usable:
                    feature = np.zeros((len(errors), 4))
                    feature[mask] = np.column_stack((np.ones(count), error[:, 0] ** 2,
                                                     error[:, 0] * error[:, 1], error[:, 1] ** 2))
                    features.append(feature)
        usable = [item for item in category_profiles if item["usable_for_gate"]]
        design = np.concatenate(features, axis=1) if features else None
        _bootstrap_items(usable, design, rng, draws, confidence)
        for item in usable:
            trace = float(np.trace(item["M_hat"]))
            item["lower"] = max(0., trace - 2 * item["radius"])
            item["upper"] = trace + 2 * item["radius"]
        profiles.extend(category_profiles)
        print(f"Motion bootstrap fitted: {category}, {int(selected.sum())} trajectories", flush=True)
    return profiles, exclusions


