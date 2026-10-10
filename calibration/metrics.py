"""Evaluate frozen group-moment envelopes on independent physical scenarios.

Empirical group moments are finite-sample diagnostics, not individual-error
coverage or a verification of the bootstrap confidence level.
"""

import numpy as np

from .bootstrap import _validate_dataset, _valid_category_points
from .motion import MOTION_TAGS


def require_disjoint_groups(certificate, rows):
    """Reject scenario leakage, including different CAV views of one scenario."""
    provenance = certificate.document.get("fit_provenance")
    fitted = provenance.get("source_group_ids") if isinstance(provenance, dict) else None
    if (not isinstance(fitted, list) or not fitted
            or any(not isinstance(group, str) or not group.strip() for group in fitted)):
        raise ValueError("Certificate needs calibration-fit physical source_group_ids for evaluation.")
    evaluated = {row["source_group_id"] for row in rows}
    overlap = set(fitted) & evaluated
    if overlap:
        raise ValueError("Calibration fit and evaluation overlap physical source groups: {}".format(
            ", ".join(sorted(overlap))))
    return sorted(evaluated)


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def _group_selections(category, step, valid, traces, tags, profiles, motion):
    """Assign independent evaluation rows using the frozen grouping rule."""
    if motion:
        for tag in MOTION_TAGS:
            profile = profiles.get((category, tag, step + 1))
            yield profile, valid[:, step] & (tags == tag), {"motion_tag": tag}
        return
    profile = profiles.get((category, None, step + 1))
    if profile is None:
        yield None, valid[:, step], {"uncertainty_bin": None}
        return
    assignments = np.searchsorted(profile["uncertainty_cutpoints"], traces[:, step], side="right")
    for item in profile["bins"]:
        yield item, valid[:, step] & (assignments == item["uncertainty_bin"]), {
            "uncertainty_bin": item["uncertainty_bin"]}


def evaluate_predictions(certificate, means, targets, covariances, categories,
                         sample_fps, *, grouping_covariances=None, slice_name="native",
                         motion_tags=None):
    """Compare group moments and mean risks using frozen bins or motion tags.

For ``wire_covariance`` the positions remain native; only covariance has passed
through the codec. Position quantization and interpolation require aligned
runtime captures and are deliberately not represented by that slice.
    """
    means, targets, covariances, categories, sample_fps = _validate_dataset(
        means, targets, covariances, categories, sample_fps)
    contract = certificate.document.get("inference", {})
    if (means.shape[1] != contract.get("pred_len")
            or not np.isclose(sample_fps, contract.get("sample_fps", 0), rtol=1e-9, atol=0)):
        raise ValueError("Evaluation horizons and sample_fps must match the frozen certificate.")
    grouping = covariances if grouping_covariances is None else np.asarray(
        grouping_covariances, dtype=np.float64)
    if grouping.shape != covariances.shape:
        raise ValueError("grouping_covariances must match reported covariance shape.")
    motion = certificate.document.get("method") == "motion_bootstrap"
    tags = np.asarray(motion_tags, dtype=object) if motion else None
    if motion and (tags.shape != (len(means),) or any(
            tag is not None and tag not in MOTION_TAGS for tag in tags)):
        raise ValueError("Motion evaluation requires one native-prediction motion tag or None per row.")
    groups, exclusions = [], {}
    certified_points, valid_points = 0, 0
    upper_sum, lower_sum, actual_sum = 0., 0., 0.
    document_profiles = {
        (profile["category"], profile.get("motion_tag"), profile["horizon_step"]): profile
        for profile in certificate.document["profiles"]
    }
    for category in sorted(set(categories)):
        mask = categories == category
        values, reported = means[mask], covariances[mask]
        errors, _, valid, exclusions[category] = _valid_category_points(
            values, targets[mask], reported)
        category_grouping = grouping[mask]
        _, grouping_traces, grouping_valid, _ = _valid_category_points(
            values, targets[mask], category_grouping)
        valid &= grouping_valid
        valid_points += int(valid.sum())
        category_tags = tags[mask] if motion else None
        if motion:
            exclusions[category]["unavailable_motion_trajectories"] = int(sum(
                tag is None for tag in category_tags))
        for step in range(means.shape[1]):
            horizon_ms = round((step + 1) * 1000 / sample_fps)
            for item, selected, identity in _group_selections(
                    category, step, valid, grouping_traces, category_tags, document_profiles, motion):
                count = int(selected.sum())
                result = {
                    "category": category, "horizon_ms": horizon_ms, **identity,
                    "n_eval": count, "certificate_available": bool(
                        item is not None and item["usable_for_gate"]),
                }
                if count:
                    group_error = errors[selected, step]
                    empirical = group_error.T @ group_error / count
                    result.update({"empirical_raw_moment": empirical.tolist(),
                                   "mean_squared_error": float(np.trace(empirical))})
                if item is None:
                    result["status"] = "missing_profile"
                    groups.append(result)
                    continue
                result["n_fit"] = item["n_valid"]
                if not item["usable_for_gate"] or not count:
                    result["status"] = "unavailable_certificate" if not item["usable_for_gate"] else "empty_evaluation_group"
                    groups.append(result)
                    continue
                upper, lower = [], []
                retained = []
                for index in np.flatnonzero(selected):
                    bounds = certificate.bounds(
                        category, horizon_ms, reported[index, step],
                        grouping_covariance=category_grouping[index, step],
                        motion_tag=identity.get("motion_tag"))
                    if bounds is None:
                        continue
                    upper.append(bounds["upper"])
                    lower.append(bounds["lower"])
                    retained.append(index)
                if not retained:
                    result["status"] = "unavailable_risk_bounds"
                    groups.append(result)
                    continue
                error = errors[retained, step]
                empirical = error.T @ error / len(retained)
                fitted = np.asarray(item["M_hat"], dtype=np.float64)
                radius = float(item["radius"])
                upper_slack = float(np.linalg.eigvalsh(fitted + radius * np.eye(2) - empirical)[0])
                lower_slack = float(np.linalg.eigvalsh(empirical - fitted + radius * np.eye(2))[0])
                tolerance = 1e-10 * max(1., np.linalg.norm(empirical, 2), np.linalg.norm(fitted, 2), radius)
                actual = float(np.trace(empirical))
                upper_mean, lower_mean = float(np.mean(upper)), float(np.mean(lower))
                result.update({
                    "status": "evaluated", "n_certified": len(retained),
                    "empirical_raw_moment": empirical.tolist(),
                    "upper_moment_holds_empirically": upper_slack >= -tolerance,
                    "lower_moment_holds_empirically": lower_slack >= -tolerance,
                    "upper_min_eigenvalue_slack": upper_slack,
                    "lower_min_eigenvalue_slack": lower_slack,
                    "mean_squared_error": actual,
                    "mean_upper_risk": upper_mean, "mean_lower_risk": lower_mean,
                    "upper_trace_holds_empirically": upper_mean >= actual - tolerance,
                    "lower_trace_holds_empirically": lower_mean <= actual + tolerance,
                    "mean_risk_width": upper_mean - lower_mean,
                    "upper_to_error_ratio": _ratio(upper_mean, actual),
                    "lower_to_error_ratio": _ratio(lower_mean, actual),
                    "group_moment_trace_width": 4 * radius,
                })
                certified_points += len(retained)
                upper_sum += float(np.sum(upper))
                lower_sum += float(np.sum(lower))
                actual_sum += actual * len(retained)
                groups.append(result)
    evaluated = [group for group in groups if group["status"] == "evaluated"]
    return {
        "slice": slice_name,
        "risk_unit": "m2",
        "grouping": "category_motion_horizon" if motion else "category_horizon_covariance_trace_bin",
        "interpretation": "empirical independent-group diagnostics; not individual-error coverage or confidence validation",
        "n_trajectories": len(means), "valid_points": valid_points,
        "certified_points": certified_points,
        "available_fraction": _ratio(certified_points, valid_points),
        "evaluated_groups": len(evaluated),
        "upper_moment_holding_groups": sum(group["upper_moment_holds_empirically"] for group in evaluated),
        "lower_moment_holding_groups": sum(group["lower_moment_holds_empirically"] for group in evaluated),
        "mean_squared_error_on_certified_points": _ratio(actual_sum, certified_points),
        "mean_upper_risk_on_certified_points": _ratio(upper_sum, certified_points),
        "mean_lower_risk_on_certified_points": _ratio(lower_sum, certified_points),
        "exclusions": exclusions, "groups": groups,
    }


def wire_covariances(covariances):
    """Reconstruct the covariance transmitted by the production codec."""
    from intelligent_vehicles.prediction_codec import pack_covariances, decode_covariances

    covariances = np.asarray(covariances, dtype=np.float64)
    decoded = np.empty_like(covariances)
    for index, trajectory in enumerate(covariances):
        values = np.column_stack((trajectory[:, 0, 0], trajectory[:, 0, 1], trajectory[:, 1, 1]))
        try:
            decoded[index] = decode_covariances(pack_covariances(values), len(trajectory))
        except ValueError:
            # The broadcaster cannot transmit this trajectory's covariance.
            decoded[index] = np.nan
    return decoded
