"""Raw-error motion certificates conditioned on native covariance-trace bins.

Fit-only quantiles define each category/motion/horizon's bins. Whole-trajectory
bootstrap draws are shared across all parents, bins and horizons in a category.
Sparse bins use their parent motion/horizon risk at lookup; this fallback is a
parent-population estimate, not a conditional-bin guarantee.
"""

import numpy as np
from scipy.sparse import csc_matrix

from .bootstrap import (
    BOOTSTRAP_DRAWS, CONFIDENCE, MIN_VALID_ROWS, SEED,
    _bootstrap_items, _valid_category_points, _validate_bootstrap_settings, _validate_dataset,
)
from .motion import MOTION_TAGS


ESTIMATOR = "motion_covariance_raw_moment"
DEFAULT_COV_BINS = 10


def validate_cov_bins(cov_bins):
    if isinstance(cov_bins, (bool, np.bool_)) or not isinstance(cov_bins, (int, np.integer)) or cov_bins < 1:
        raise ValueError("cov_bins must be a positive integer.")


def _item(errors, mask, minimum, **fields):
    count = int(mask.sum())
    usable = count >= minimum
    error = errors[mask]
    return dict(fields, n_valid=count, usable_for_gate=usable,
                M_hat=(error.T @ error / count).tolist() if usable else None,
                radius=None, lower=None, upper=None,
                status="pending" if usable else "insufficient_fit_samples")


def estimate_motion_covariance_profiles(
        means, targets, covariances, categories, motion_tags, sample_fps, *,
        cov_bins=DEFAULT_COV_BINS, draws=BOOTSTRAP_DRAWS, seed=SEED,
        min_rows=MIN_VALID_ROWS, confidence=CONFIDENCE):
    """Fit raw moments and m² risks; covariance selects bins without scaling risk."""
    validate_cov_bins(cov_bins)
    means, targets, covariances, categories, sample_fps = _validate_dataset(
        means, targets, covariances, categories, sample_fps)
    _validate_bootstrap_settings(draws, seed, min_rows, confidence)
    tags = np.asarray(motion_tags, dtype=object)
    if tags.shape != (len(means),) or any(tag is not None and tag not in MOTION_TAGS for tag in tags):
        raise ValueError("motion_tags must contain one known tag or None per trajectory.")
    quantiles = np.arange(1, cov_bins, dtype=float) / cov_bins
    profiles, exclusions = [], {}
    rng = np.random.default_rng(seed)
    for category in sorted(set(categories)):
        design = None
        selected = categories == category
        category_tags = tags[selected]
        errors, traces, valid, exclusions[category] = _valid_category_points(
            means[selected], targets[selected], covariances[selected])
        exclusions[category]["unavailable_motion_trajectories"] = int(sum(tag is None for tag in category_tags))
        category_profiles, usable, data, rows, indptr = [], [], [], [], [0]
        row_dtype = np.int32 if len(errors) <= np.iinfo(np.int32).max else np.int64

        def register(item, point_errors, mask):
            if not item["usable_for_gate"]:
                return
            indices = np.flatnonzero(mask).astype(row_dtype, copy=False)
            e = point_errors[mask]
            stats = np.column_stack((np.ones(len(e)), e[:, 0] ** 2, e[:, 0] * e[:, 1], e[:, 1] ** 2))
            data.append(stats.T.ravel())
            rows.append(np.tile(indices, 4))
            indptr.extend(range(indptr[-1] + len(e), indptr[-1] + 5 * len(e), len(e)))
            usable.append(item)

        for tag in MOTION_TAGS:
            for step in range(means.shape[1]):
                mask = valid[:, step] & (category_tags == tag)
                point_errors, point_traces = errors[:, step], traces[:, step]
                parent = _item(point_errors, mask, min_rows, category=category, motion_tag=tag,
                               horizon_step=step + 1, horizon_seconds=(step + 1) / sample_fps)
                cutpoints = np.unique(np.quantile(point_traces[mask], quantiles)) if mask.any() else np.empty(0)
                assignments = np.searchsorted(cutpoints, point_traces, side="right")
                parent.update(uncertainty_cutpoints=cutpoints.tolist(), bins=[])
                register(parent, point_errors, mask)
                for index in range(len(cutpoints) + 1):
                    item = _item(point_errors, mask & (assignments == index), min_rows,
                                 uncertainty_bin=index)
                    parent["bins"].append(item)
                    register(item, point_errors, mask & (assignments == index))
                category_profiles.append(parent)
        if usable:
            design = csc_matrix((np.concatenate(data), np.concatenate(rows), indptr),
                                shape=(len(errors), 4 * len(usable)))
            data.clear()
            rows.clear()
            _bootstrap_items(usable, design, rng, draws, confidence)
            for item in usable:
                trace, radius = float(np.trace(item["M_hat"])), item["radius"]
                item.update(lower=max(0., trace - 2 * radius), upper=trace + 2 * radius)
        profiles.extend(category_profiles)
        print("Motion covariance bootstrap fitted: {}, {} trajectories, {} supported moments".format(
            category, int(selected.sum()), len(usable)), flush=True)
    return profiles, exclusions
