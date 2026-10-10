"""Load one predictor's certificate and select native expected-error risks.

Group assignment uses the original model covariance. Factor conversion may
use the decoded wire covariance, keeping transmitted factors paired with P.
Motion certificates directly store squared-error risks in square metres.
Neither method establishes certificate transfer to interpolated means.
"""

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from calibration.motion import MOTION_DEFINITION, MOTION_TAGS


def file_identity(path):
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def valid_bounds(bounds):
    """Unavailable bounds are None; malformed non-null bounds are invalid."""
    if not isinstance(bounds, Mapping):
        return False
    values = [bounds.get("lower"), bounds.get("upper")]
    return (all(isinstance(v, (int, float, np.integer, np.floating))
                and not isinstance(v, (bool, np.bool_)) and np.isfinite(v)
                for v in values) and 0 <= values[0] <= values[1])


def valid_factors(factors):
    """Validate dimensionless factors used internally by bootstrap profiles."""
    return (isinstance(factors, Mapping) and valid_bounds({
        "lower": factors.get("ell"), "upper": factors.get("u"),
    }))


def _number(value, path, minimum=0, inclusive=False, integer=False):
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if (not valid_type or isinstance(value, bool) or not math.isfinite(value)
            or (value < minimum if inclusive else value <= minimum)):
        raise ValueError(f"{path}: invalid finite {'integer' if integer else 'number'}")


def _boolean(value, path):
    if not isinstance(value, bool):
        raise ValueError(f"{path}: must be a boolean")


def _covariance(value):
    covariance = np.asarray(value, dtype=np.float64)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("Covariance must be a finite 2x2 matrix.")
    tolerance = 1e-12 + 1e-10 * np.max(np.abs(covariance))
    if np.max(np.abs(covariance - covariance.T)) > tolerance:
        raise ValueError("Covariance must be symmetric.")
    covariance = 0.5 * covariance + 0.5 * covariance.T
    np.linalg.cholesky(covariance)
    return covariance


class Certificate:
    """A validated local artifact; receivers need only the selected L/U risks."""

    def __init__(self, document):
        if not isinstance(document, dict):
            raise ValueError("Certificate must be a dictionary.")
        self.document = document
        self.inference = document.get("inference")
        if not isinstance(self.inference, dict):
            raise ValueError("Certificate inference must be a dictionary.")
        _number(self.inference.get("pred_len"), "inference.pred_len", integer=True)
        _number(self.inference.get("sample_fps"), "inference.sample_fps")
        self.method = document.get("estimator")
        if self.method == "motion_bootstrap_raw_moment":
            self._load_motion_profiles(document)
        else:
            self.profiles = _certificate_profiles(document, "certificate", self.inference)
        self.certificate_id = hashlib.sha256(json.dumps(
            document, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()

    @classmethod
    def from_file(cls, path, checkpoint_path=None, expected=None):
        path = Path(path).expanduser().resolve()
        try:
            raw = path.read_bytes()
            certificate = cls(json.loads(raw))
            if checkpoint_path is not None:
                identity = certificate.document.get("checkpoint", {})
                if not isinstance(identity, dict):
                    raise ValueError("Certificate checkpoint identity must be a dictionary.")
                if identity.get("sha256") != file_identity(checkpoint_path)["sha256"]:
                    raise ValueError("Checkpoint SHA256 does not match the certificate.")
            for name, value in (expected or {}).items():
                actual = certificate.inference.get(name)
                if name == "sample_fps":
                    _number(actual, "inference.sample_fps")
                    matches = math.isclose(actual, value, rel_tol=1e-9, abs_tol=0)
                else:
                    matches = type(actual) is type(value) and actual == value
                if not matches:
                    raise ValueError(f"inference.{name}: expected {value!r}, got {actual!r}")
            certificate.certificate_id = hashlib.sha256(raw).hexdigest()
            return certificate
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error

    def factors(self, category, horizon_ms, covariance, grouping_covariance=None):
        """Return native-step factors or None for unavailable profiles/bins/P."""
        if self.method != "bootstrap_raw_moment":
            return None
        if not isinstance(horizon_ms, (int, np.integer)) or isinstance(horizon_ms, (bool, np.bool_)):
            return None
        if not isinstance(category, str):
            return None
        profile = self.profiles.get(category, {}).get(int(horizon_ms))
        if profile is None:
            return None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                covariance = _covariance(covariance)
                group_covariance = (covariance if grouping_covariance is None
                                    else _covariance(grouping_covariance))
                index = int(np.searchsorted(profile["cutpoints"],
                                            np.trace(group_covariance), side="right"))
                bounds = profile["bins"][index]
                if bounds is None:
                    return None
                moment = np.asarray(bounds["M_hat"], dtype=np.float64)
                radius = bounds["radius"] * np.eye(2)
                inverse = np.linalg.solve(np.linalg.cholesky(covariance), np.eye(2))
                lower = inverse @ (moment - radius) @ inverse.T
                upper = inverse @ (moment + radius) @ inverse.T
                result = {
                    "ell": max(0., float(np.linalg.eigvalsh(0.5 * lower + 0.5 * lower.T)[0])),
                    "u": float(np.linalg.eigvalsh(0.5 * upper + 0.5 * upper.T)[-1]),
                }
                return result if valid_factors(result) else None
        except (ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
            return None

    def bounds(self, category, horizon_ms, covariance=None, grouping_covariance=None,
               motion_tag=None):
        """Return native expected squared-error bounds in m², or None."""
        if self.method == "motion_bootstrap_raw_moment":
            return self.trajectory_bounds(category, [horizon_ms], None, motion_tag)[0]
        factors = self.factors(category, horizon_ms, covariance, grouping_covariance)
        if factors is None:
            return None
        try:
            with np.errstate(over="raise", invalid="raise"):
                trace = float(np.trace(_covariance(covariance)))
                result = {"lower": factors["ell"] * trace,
                          "upper": factors["u"] * trace}
                return result if valid_bounds(result) else None
        except (ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
            return None

    def trajectory_bounds(self, category, horizons_ms, covariances=None, motion_tag=None):
        """Select one motion row, then retrieve bounds for all native horizons."""
        horizons = list(horizons_ms)
        if self.method != "motion_bootstrap_raw_moment":
            if covariances is None:
                return [None] * len(horizons)
            covariances = list(covariances)
            if len(covariances) != len(horizons):
                raise ValueError("Covariances must match the native horizon count.")
            return [self.bounds(category, horizon, covariance)
                    for horizon, covariance in zip(horizons, covariances)]
        category_index = self._category_index.get(category) if isinstance(category, str) else None
        tag_index = self._motion_index.get(motion_tag) if isinstance(motion_tag, str) else None
        if category_index is None or tag_index is None:
            return [None] * len(horizons)
        lower = self._lower[category_index, tag_index]
        upper = self._upper[category_index, tag_index]
        available = self._available[category_index, tag_index]
        result = []
        for horizon in horizons:
            index = (self._horizon_index.get(int(horizon))
                     if isinstance(horizon, (int, np.integer))
                     and not isinstance(horizon, (bool, np.bool_)) else None)
            result.append({"lower": float(lower[index]), "upper": float(upper[index])}
                          if index is not None and available[index] else None)
        return result

    def _load_motion_profiles(self, document):
        """Validate the frozen complete table before constructing dense lookup arrays."""
        if type(document.get("schema_version")) is not int or document["schema_version"] != 3:
            raise ValueError("Motion certificates require schema_version=3.")
        if document.get("method") != "motion_bootstrap" or document.get("risk_unit") != "m2":
            raise ValueError("Motion certificates require method=motion_bootstrap and risk_unit=m2.")
        if document.get("motion") != MOTION_DEFINITION:
            raise ValueError("Certificate motion definition differs from the runtime categorizer.")
        _number(self.inference.get("obs_len"), "inference.obs_len", integer=True)
        if self.inference.get("coordinate_convention") != "waymo_v1":
            raise ValueError("Motion certificates require Waymo coordinates.")
        profiles = document.get("profiles")
        if not isinstance(profiles, list) or not profiles:
            raise ValueError("Motion certificate profiles must be a non-empty list.")
        categories = set()
        for profile in profiles:
            category = profile.get("category") if isinstance(profile, dict) else None
            if not isinstance(category, str) or not category.strip():
                raise ValueError("Each motion profile requires a non-empty category.")
            categories.add(category)
        self._category_index = {category: index for index, category in enumerate(sorted(categories))}
        self._motion_index = {tag: index for index, tag in enumerate(MOTION_TAGS)}
        pred_len = self.inference["pred_len"]
        horizons = [round(1000 * step / self.inference["sample_fps"])
                    for step in range(1, pred_len + 1)]
        if horizons[0] <= 0 or len(set(horizons)) != pred_len:
            raise ValueError("Native horizons must be distinct positive milliseconds.")
        self._horizon_index = {horizon: index for index, horizon in enumerate(horizons)}
        shape = (len(categories), len(MOTION_TAGS), pred_len)
        self._lower = np.zeros(shape, dtype=np.float64)
        self._upper = np.zeros(shape, dtype=np.float64)
        self._available = np.zeros(shape, dtype=bool)
        settings = document.get("settings")
        if not isinstance(settings, dict):
            raise ValueError("Motion certificate settings must be a dictionary.")
        minimum = settings.get("min_fit_rows")
        _number(minimum, "settings.min_fit_rows", integer=True)
        seen = set()
        for index, profile in enumerate(profiles):
            path = f"certificate.profiles[{index}]"
            tag = profile.get("motion_tag")
            if not isinstance(tag, str) or tag not in self._motion_index:
                raise ValueError(f"{path}.motion_tag: unknown motion category")
            step = profile.get("horizon_step")
            _number(step, path + ".horizon_step", integer=True)
            if step > pred_len:
                raise ValueError(f"{path}.horizon_step: exceeds prediction length")
            seconds = profile.get("horizon_seconds")
            _number(seconds, path + ".horizon_seconds")
            if not math.isclose(seconds, step / self.inference["sample_fps"],
                                rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError(f"{path}.horizon_seconds: inconsistent with native step")
            key = (self._category_index[profile["category"]], self._motion_index[tag], step - 1)
            if key in seen:
                raise ValueError(f"{path}: duplicate category/motion/horizon")
            seen.add(key)
            count = profile.get("n_valid")
            _number(count, path + ".n_valid", integer=True, inclusive=True)
            usable = profile.get("usable_for_gate")
            _boolean(usable, path + ".usable_for_gate")
            if not usable:
                if (count >= minimum or profile.get("status") != "insufficient_fit_samples"
                        or any(profile.get(name) is not None
                               for name in ("M_hat", "radius", "lower", "upper"))):
                    raise ValueError(f"{path}: inconsistent unavailable profile")
                continue
            if count < minimum or profile.get("status") != "approximate_group_moment":
                raise ValueError(f"{path}: usable profile lacks minimum support or valid status")
            moment = _moment(profile.get("M_hat"), path + ".M_hat")
            radius = profile.get("radius")
            _number(radius, path + ".radius", inclusive=True)
            bounds = {"lower": profile.get("lower"), "upper": profile.get("upper")}
            if not valid_bounds(bounds):
                raise ValueError(f"{path}: invalid squared-error bounds")
            trace = float(np.trace(moment))
            expected = {"lower": max(0., trace - 2 * radius), "upper": trace + 2 * radius}
            if any(not math.isclose(bounds[name], expected[name], rel_tol=1e-10, abs_tol=1e-12)
                   for name in expected):
                raise ValueError(f"{path}: bounds disagree with moment and radius")
            self._lower[key], self._upper[key] = bounds["lower"], bounds["upper"]
            self._available[key] = True
        if len(seen) != np.prod(shape):
            raise ValueError("Motion profiles must cover every category, motion tag and native horizon.")
        usable_count = document.get("usable_profile_count")
        _number(usable_count, "usable_profile_count", integer=True, inclusive=True)
        if usable_count != int(self._available.sum()):
            raise ValueError("usable_profile_count disagrees with motion profiles.")
        self.profiles = {}


def _moment(raw, path):
    if (not isinstance(raw, list) or len(raw) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in raw)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for row in raw for value in row)):
        raise ValueError(f"{path}: must be a finite numeric 2x2 matrix")
    moment = np.asarray(raw, dtype=np.float64)
    tolerance = 1e-12 + 1e-10 * np.max(np.abs(moment))
    if np.max(np.abs(moment - moment.T)) > tolerance:
        raise ValueError(f"{path}: must be symmetric")
    moment = 0.5 * moment + 0.5 * moment.T
    if np.linalg.eigvalsh(moment)[0] < -tolerance:
        raise ValueError(f"{path}: must be positive semidefinite")
    return moment


def _bootstrap_profile(profile, path):
    """Validate and index one horizon's frozen uncertainty bins."""
    cutpoints = profile.get("uncertainty_cutpoints")
    if not isinstance(cutpoints, list):
        raise ValueError(f"{path}.uncertainty_cutpoints: must be a list")
    for index, value in enumerate(cutpoints):
        _number(value, f"{path}.uncertainty_cutpoints[{index}]", inclusive=True)
        if index and value <= cutpoints[index - 1]:
            raise ValueError(f"{path}.uncertainty_cutpoints: must be strictly increasing")
    bins = profile.get("bins")
    if not isinstance(bins, list) or len(bins) != len(cutpoints) + 1:
        raise ValueError(f"{path}.bins: requires one bin per interval")
    indexed = []
    for index, entry in enumerate(bins):
        location = f"{path}.bins[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{location}: must be a dictionary")
        bin_index = entry.get("uncertainty_bin")
        _number(bin_index, location + ".uncertainty_bin", integer=True, inclusive=True)
        if bin_index != index:
            raise ValueError(f"{location}.uncertainty_bin: bins must be complete and ordered")
        _number(entry.get("n_valid"), location + ".n_valid", integer=True, inclusive=True)
        usable = entry.get("usable_for_gate")
        _boolean(usable, location + ".usable_for_gate")
        if not usable:
            if entry.get("M_hat") is not None or entry.get("radius") is not None:
                raise ValueError(f"{location}: unavailable bins must have null moment and radius")
            indexed.append(None)
            continue
        if entry["n_valid"] == 0:
            raise ValueError(f"{location}.n_valid: usable bins require observations")
        _number(entry.get("radius"), location + ".radius", inclusive=True)
        moment = _moment(entry.get("M_hat"), location + ".M_hat")
        indexed.append({"M_hat": moment.tolist(), "radius": float(entry["radius"])})
    return {"method": "bootstrap_raw_moment", "cutpoints": cutpoints, "bins": indexed}


def _certificate_profiles(certificate, path, inference):
    """Index native-point profiles; unavailable bins stay explicitly absent."""
    if type(certificate.get("schema_version")) is not int or certificate["schema_version"] != 2:
        raise ValueError(f"{path}: requires bootstrap certificate schema_version=2")
    if certificate.get("estimator") != "bootstrap_raw_moment":
        raise ValueError(f"{path}.estimator: requires bootstrap_raw_moment")
    profiles = certificate.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(f"{path}.profiles: must be a non-empty list")
    indexed, seen = {}, set()
    for index, profile in enumerate(profiles):
        location = f"{path}.profiles[{index}]"
        if not isinstance(profile, dict):
            raise ValueError(f"{location}: must be a dictionary")
        category = profile.get("category")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{location}.category: must be a non-empty string")
        step = profile.get("horizon_step")
        _number(step, location + ".horizon_step", integer=True)
        if step > inference["pred_len"]:
            raise ValueError(f"{location}.horizon_step: exceeds prediction length")
        seconds = profile.get("horizon_seconds")
        _number(seconds, location + ".horizon_seconds")
        if not math.isclose(seconds, step / inference["sample_fps"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{location}.horizon_seconds: inconsistent with horizon_step and sample_fps")
        horizon_ms = round(seconds * 1000)
        if horizon_ms <= 0 or (category, horizon_ms) in seen:
            raise ValueError(f"{location}: duplicate or invalid category/horizon in milliseconds")
        seen.add((category, horizon_ms))
        bounds = _bootstrap_profile(profile, location)
        if any(value is not None for value in bounds["bins"]):
            indexed.setdefault(category, {})[horizon_ms] = bounds
    usable_count = sum(len(entries) for entries in indexed.values())
    if "usable_profile_count" in certificate:
        _number(certificate["usable_profile_count"], path + ".usable_profile_count",
                integer=True, inclusive=True)
        if certificate["usable_profile_count"] != usable_count:
            raise ValueError(f"{path}.usable_profile_count: inconsistent with profiles")
    return indexed


