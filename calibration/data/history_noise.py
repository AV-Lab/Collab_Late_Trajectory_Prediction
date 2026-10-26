"""Category/observed-length Gaussian models of measured XY history errors.

Each draw retains the measured bias and full temporal/XY covariance. Leading
missing states are padding; agents are sampled independently. Singular support
is retained without covariance floors or jitter.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np


MODEL_TYPE = "joint_gaussian_history_residual"
COORDINATE_CONVENTION = "fixed per-history heading; x forward, y left"


def _rotate(values, heading, inverse):
    values, heading = np.asarray(values, dtype=float), np.asarray(heading, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError("Vectors must end in two XY coordinates")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(heading)):
        raise ValueError("Rotation inputs must be finite")
    cosine, sine = np.cos(heading), np.sin(heading)
    while cosine.ndim < values.ndim - 1:
        cosine, sine = cosine[..., None], sine[..., None]
    x, y = values[..., 0], values[..., 1]
    if inverse:
        return np.stack((cosine * x - sine * y, sine * x + cosine * y), axis=-1)
    return np.stack((cosine * x + sine * y, -sine * x + cosine * y), axis=-1)


def world_to_local(residuals, heading):
    """Rotate world XY vectors into forward/left axes at heading radians."""
    return _rotate(residuals, heading, False)


def local_to_world(residuals, heading):
    """Invert :func:`world_to_local`; heading can be scalar or per history."""
    return _rotate(residuals, heading, True)


def _strings(values, count, name):
    result = np.asarray(values)
    if result.shape != (count,) or any(value is None for value in result):
        raise ValueError(f"{name} must contain one nonempty value per history")
    result = result.astype(str, copy=True)
    if np.any(result == ""):
        raise ValueError(f"{name} must not contain empty values")
    return result


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _frequency(value):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError("sample_fps must be positive and finite")
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("sample_fps must be positive and finite") from error
    if not np.isfinite(value) or value <= 0:
        raise ValueError("sample_fps must be positive and finite")
    return value


def _lengths(values, count, history_steps):
    values = np.asarray(values)
    if (values.shape != (count,) or not np.issubdtype(values.dtype, np.number)
            or not np.all(np.isfinite(values)) or np.any(values < 1)
            or np.any(values != np.floor(values)) or np.any(values > np.iinfo(np.int64).max)):
        raise ValueError("history_lengths must be a vector of positive finite integers")
    return np.minimum(values, history_steps).astype(np.int64)


def _psd_eigh(covariance):
    """Use the fitted PSD support; remove only machine-precision null values."""
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not covariance.size or not np.all(np.isfinite(covariance))):
        raise ValueError("Covariance must be a finite nonempty square matrix")
    tolerance = np.finfo(float).eps * len(covariance) * np.max(np.abs(covariance)) * 8
    if np.max(np.abs(covariance - covariance.T)) > tolerance:
        raise ValueError("Covariance is not symmetric within floating-point precision")
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    tolerance = np.finfo(float).eps * len(covariance) * np.max(np.abs(values)) * 8
    if np.any(values < -tolerance):
        raise ValueError("Covariance has a materially negative eigenvalue")
    values[np.abs(values) <= tolerance] = 0.
    return covariance, values, vectors, float(tolerance)


def _profile(category, length, mean, covariance, counts, history_steps, minimum):
    mean = np.asarray(mean, dtype=float)
    covariance, values, vectors, tolerance = _psd_eigh(covariance)
    if (mean.shape != (2 * length,) or not np.all(np.isfinite(mean))
            or covariance.shape != (2 * length, 2 * length)):
        raise ValueError("Profile mean/covariance must have shapes (2k,) and (2k, 2k)")
    description = dict(counts, category=str(category), history_length=int(length),
        supported=counts["n_histories"] >= minimum,
        reason="" if counts["n_histories"] >= minimum else "sparse_pool",
        valid_timestep_indices=list(range(history_steps - length, history_steps)),
        mean_valid_2k=mean.tolist(), mean_valid_kx2=mean.reshape(length, 2).tolist(),
        covariance_valid_2kx2k=covariance.tolist(),
        joint_rank=int(np.count_nonzero(values > 0)),
        joint_eigenvalues=values.tolist(), eigenvalue_roundoff_tolerance=tolerance)
    result = dict(mean=mean.copy(), covariance=covariance, factor=vectors * np.sqrt(values)[None, :],
                  description=description)
    for name in ("mean", "covariance", "factor"):
        result[name].flags.writeable = False
    return result


class HistoryNoiseModel:
    """Compact fitted parameters; measured donor histories are not retained."""

    def __init__(self, profiles, min_pool_size, history_steps, sample_fps, metadata=None):
        self._profiles = profiles
        self.min_pool_size = _positive_integer(min_pool_size, "min_pool_size", 2)
        self.history_steps = _positive_integer(history_steps, "history_steps")
        self.sample_fps = _frequency(sample_fps)
        self.coordinate_convention = COORDINATE_CONVENTION
        self._fit_metadata = copy.deepcopy(metadata or {})

    @property
    def fit_metadata(self):
        return copy.deepcopy(self._fit_metadata)

    def summary(self):
        pools = [copy.deepcopy(value["description"]) for _, value in sorted(self._profiles.items())]
        probabilities = []
        for category in sorted({key[0] for key in self._profiles}):
            selected = [pool for pool in pools if pool["category"] == category]
            total = sum(pool["n_histories"] for pool in selected)
            supported = sum(pool["n_histories"] for pool in selected if pool["supported"])
            probabilities.extend(dict(category=category, history_length=pool["history_length"],
                n_histories=pool["n_histories"], supported=pool["supported"],
                calibration_probability=pool["n_histories"] / total,
                sampling_probability=pool["n_histories"] / supported if pool["supported"] and supported else 0.)
                for pool in selected)
        return dict(schema_version=2, model_type=MODEL_TYPE, history_steps=self.history_steps,
            sample_fps=self.sample_fps, coordinate_convention=self.coordinate_convention,
            min_pool_size=self.min_pool_size, covariance_denominator="N-1",
            variance_floor=None, covariance_jitter=0., same_scene_conditioning=False,
            distributional_assumption="joint Gaussian per category and observed history length",
            sampling_unit="independent fresh joint XY residual history per agent",
            future_coverage_guarantee=False, no_donor_bank_required_after_loading=True,
            n_histories=sum(pool["n_histories"] for pool in pools), n_pools=len(pools),
            n_supported_pools=sum(pool["supported"] for pool in pools),
            fit_metadata=self.fit_metadata, length_probabilities=probabilities, pools=pools)

    def save(self, path):
        document = json.dumps(self.summary(), indent=2, allow_nan=False) + "\n"
        with Path(path).open("x") as stream:
            stream.write(document)

    @classmethod
    def from_file(cls, path):
        """Load saved parameters; schema 1 explicitly uses its legacy 10 Hz grid."""
        payload = json.loads(Path(path).read_text())
        version = payload.get("schema_version")
        if payload.get("model_type") != MODEL_TYPE or version not in (1, 2):
            raise ValueError("Unsupported history-noise model format")
        if payload.get("covariance_denominator") != "N-1":
            raise ValueError("Unsupported covariance convention")
        steps = _positive_integer(payload.get("history_steps", 10 if version == 1 else None), "history_steps")
        if version == 1 and steps != 10 and "sample_fps" not in payload:
            raise ValueError("Legacy metadata defaults apply only to ten-step profiles")
        fps = _frequency(payload.get("sample_fps", 10. if version == 1 else None))
        if payload.get("coordinate_convention", COORDINATE_CONVENTION) != COORDINATE_CONVENTION:
            raise ValueError("Unsupported residual coordinate convention")
        minimum = _positive_integer(payload["min_pool_size"], "min_pool_size", 2)
        profiles = {}
        for pool in payload["pools"]:
            category = str(pool["category"])
            length = _positive_integer(pool["history_length"], "history_length")
            if not category or length > steps or (category, length) in profiles:
                raise ValueError("Invalid or duplicate category/length profile")
            counts = {name: _positive_integer(pool[name], name)
                      for name in ("n_histories", "n_scenes", "n_tracks")}
            profiles[category, length] = _profile(category, length, pool["mean_valid_2k"],
                pool["covariance_valid_2kx2k"], counts, steps, minimum)
        if not profiles:
            raise ValueError("Saved model has no profiles")
        return cls(profiles, minimum, steps, fps, payload.get("fit_metadata"))

    load = from_file

    def sample(self, categories, history_lengths=None, *, rng=None, seed=None):
        """Draw category/length mixtures; unsupported requested pools have no fallback."""
        raw = np.asarray(categories)
        if raw.ndim != 1:
            raise ValueError("categories must be a vector")
        count, steps = len(raw), self.history_steps
        categories = _strings(raw, count, "categories")
        lengths = None if history_lengths is None else _lengths(history_lengths, count, steps)
        if rng is not None and seed is not None:
            raise ValueError("Supply either rng or seed")
        generator = np.random.default_rng(seed) if rng is None else rng
        if not isinstance(generator, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        result = dict(residuals=np.zeros((count, steps, 2)), valid_mask=np.zeros((count, steps), bool),
            available=np.zeros(count, bool), reasons=np.full(count, "", object),
            sampled_history_lengths=np.zeros(count, np.int64), profile_keys=np.full(count, "", object))
        choices = {}
        for row, category in enumerate(categories):
            if lengths is None:
                if category not in choices:
                    options = [(length, profile["description"]["n_histories"])
                               for (cat, length), profile in sorted(self._profiles.items())
                               if cat == category and profile["description"]["supported"]]
                    weights = np.asarray([number for _, number in options], dtype=float)
                    choices[category] = [length for length, _ in options], weights / weights.sum() if len(weights) else weights
                options, weights = choices[category]
                if not options:
                    result["reasons"][row] = ("no_supported_length_pool" if any(cat == category for cat, _ in self._profiles)
                                               else "missing_category_pool")
                    continue
                length = int(generator.choice(options, p=weights))
            else:
                length = int(lengths[row])
            profile = self._profiles.get((str(category), length))
            if profile is None or not profile["description"]["supported"]:
                result["reasons"][row] = "missing_category_length_pool" if profile is None else "sparse_pool"
                continue
            value = profile["mean"] + profile["factor"] @ generator.standard_normal(2 * length)
            result["residuals"][row, -length:] = value.reshape(length, 2)
            result["valid_mask"][row, -length:] = True
            result["available"][row] = True
            result["sampled_history_lengths"][row] = length
            result["profile_keys"][row] = f"{category}|{length}"
        return result


def fit_history_noise(residuals, categories, lengths, *, scene_ids, track_ids=None,
                      keys=None, sample_fps, min_pool_size=2, metadata=None):
    """Fit biased means and full N-1 covariance within category × observed length."""
    minimum = _positive_integer(min_pool_size, "min_pool_size", 2)
    fps = _frequency(sample_fps)
    raw = np.asarray(residuals, dtype=float)
    if raw.ndim != 3 or raw.shape[2] != 2 or not raw.shape[1] or not len(raw):
        raise ValueError("A nonempty residual array with shape (N, H, 2) is required")
    count, steps = raw.shape[:2]
    categories, scenes = _strings(categories, count, "categories"), _strings(scene_ids, count, "scene_ids")
    tracks = _strings(track_ids if track_ids is not None else np.arange(count), count, "track_ids")
    windows = _strings(keys if keys is not None else np.arange(count), count, "keys")
    lengths = _lengths(lengths, count, steps)
    valid = np.arange(steps)[None, :] >= steps - lengths[:, None]
    if not np.all(np.isfinite(raw[valid])):
        raise ValueError("Observed residual tails must be finite")
    if len(set(zip(scenes.tolist(), tracks.tolist(), windows.tolist()))) != count:
        raise ValueError("Duplicate scene/track/window identities would reweight the fit")
    profiles = {}
    for category, length in sorted(set(zip(categories.tolist(), lengths.tolist()))):
        indices = np.flatnonzero((categories == category) & (lengths == length))
        values = raw[indices, -length:].reshape(len(indices), 2 * length)
        mean = values[0] + np.mean(values - values[0], axis=0)
        centered = values - mean
        covariance = centered.T @ centered / (len(indices) - 1) if len(indices) > 1 else np.zeros((2 * length, 2 * length))
        counts = dict(n_histories=int(len(indices)), n_scenes=len(set(scenes[indices].tolist())),
                      n_tracks=len(set(zip(scenes[indices].tolist(), tracks[indices].tolist()))))
        profiles[category, length] = _profile(category, length, mean, covariance, counts, steps, minimum)
    fit_metadata = copy.deepcopy(metadata or {})
    fit_metadata.update(n_scenes=len(set(scenes.tolist())), source_group_ids=sorted(set(scenes.tolist())),
        n_tracks=len(set(zip(scenes.tolist(), tracks.tolist()))), coordinate_frame=COORDINATE_CONVENTION,
        donor_histories_retained=False)
    return HistoryNoiseModel(profiles, minimum, steps, fps, fit_metadata)
