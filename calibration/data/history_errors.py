"""Fit history noise from saved, identity-matched clean/tracker contexts.

Records contain ``contexts.clean`` and ``contexts.tracked`` track lists, each
with id/category/history (rows: x, y, z, heading). ``matches`` maps tracked IDs
to clean IDs; legacy ``targets`` dictionaries with clean_id/key also work.
Future GT is neither required nor read. Optional history_timestamps_s (or _ms)
on both tracks establish explicit pairing; otherwise the capture's uniform
history-grid contract must describe queues ending at the current frame.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np

from .history_noise import _frequency, _positive_integer, fit_history_noise, world_to_local
from ..runtime import file_identity


def iter_history_records(path):
    """Read a trusted local capture's gzip pickle stream or JSON/JSONL frames."""
    path = Path(path)
    if path.suffix == ".json":
        yield from json.loads(path.read_text())
    elif path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as stream:
            while True:
                try:
                    yield pickle.load(stream)
                except EOFError:
                    return


def _history_times(track, history, sample_fps):
    for name, scale in (("history_timestamps_s", 1.), ("history_timestamps_ms", .001)):
        if name in track:
            values = np.asarray(track[name], dtype=float) * scale
            if values.shape != (len(history),) or not np.all(np.isfinite(values)):
                raise ValueError("History timestamps must be finite and match history length")
            difference = np.diff(values)
            if np.any(difference <= 0) or (sample_fps is not None and not np.allclose(
                    difference, 1. / sample_fps, rtol=0., atol=2e-6)):
                raise ValueError("History timestamps disagree with the sampling grid")
            return values
    return None


def collect_history_errors(records, history_steps, *, sample_fps=None):
    """Return residual arrays and counts without changing any supplied record.

    Matches must reference tracks in the same frame. Short tracked tails retain
    a leading missing mask, and clean queues must span the requested window.
    """
    steps = _positive_integer(history_steps, "history_steps")
    fps = None if sample_fps is None else _frequency(sample_fps)
    names = ("residuals", "clean_xy", "tracked_xy", "valid_mask", "headings", "categories",
             "history_lengths", "scene_ids", "agent_ids", "track_ids", "window_keys")
    arrays, identities, frames, timestamped = {name: [] for name in names}, set(), 0, 0
    for frame in records:
        frames += 1
        contexts = frame["contexts"]
        clean = {str(track["id"]): track for track in contexts["clean"]}
        tracked = {str(track["id"]): track for track in contexts["tracked"]}
        if len(clean) != len(contexts["clean"]) or len(tracked) != len(contexts["tracked"]):
            raise ValueError("Duplicate track IDs in a paired context")
        scene = str(frame.get("scenario", frame.get("scene_id", "")))
        agent = str(frame.get("agent", frame.get("agent_id", "0")))
        if not scene:
            raise ValueError("Paired contexts require a physical scene ID")
        matches = frame.get("matches", frame.get("targets"))
        if matches is None:
            raise ValueError("Paired contexts require explicit tracked-to-clean matches")
        for tracked_id, match in matches.items():
            tracked_id = str(tracked_id)
            target = match if isinstance(match, dict) else {"clean_id": match}
            clean_id = str(target["clean_id"])
            real, truth = tracked[tracked_id], clean[clean_id]
            c, t = np.asarray(truth["history"], dtype=float), np.asarray(real["history"], dtype=float)
            if (c.ndim != 2 or t.ndim != 2 or c.shape[1] < 4 or t.shape[1] < 2
                    or len(c) < steps or not len(t) or not np.all(np.isfinite(c))
                    or not np.all(np.isfinite(t))):
                raise ValueError("Malformed matched history or incomplete clean history")
            category = str(real["category"])
            if not category or ("category" in truth and str(truth["category"]) != category):
                raise ValueError("Matched clean/tracked categories disagree")
            k = min(steps, len(t))
            clean_times, tracked_times = _history_times(truth, c, fps), _history_times(real, t, fps)
            if (clean_times is None) != (tracked_times is None):
                raise ValueError("Both matched histories must supply timestamps when either does")
            if clean_times is not None:
                timestamped += 1
                if not np.allclose(clean_times[-k:], tracked_times[-k:], rtol=0., atol=2e-6):
                    raise ValueError("Matched clean/tracked history timestamps differ")
                if "frame_time_s" in frame and not np.isclose(clean_times[-1], float(frame["frame_time_s"]), rtol=0., atol=2e-6):
                    raise ValueError("History timestamps do not end at the current frame")
            origin = frame.get("origin_ms", frame.get("frame_time_s", frame.get("frame_index", frames)))
            key = str(target.get("key", f"{scene}|{agent}|{origin}|{clean_id}"))
            identity = (scene, agent, tracked_id, key)
            if identity in identities:
                raise ValueError("Duplicate matched scene/agent/track/window identity")
            identities.add(identity)
            delta, observed = np.zeros((steps, 2)), np.zeros((steps, 2))
            heading = float(c[-1, 3])
            delta[-k:] = world_to_local(t[-k:, :2] - c[-k:, :2], heading)
            observed[-k:] = t[-k:, :2]
            values = dict(residuals=delta, clean_xy=c[-steps:, :2].copy(), tracked_xy=observed,
                valid_mask=np.arange(steps) >= steps - k, headings=heading, categories=category,
                history_lengths=k, scene_ids=scene, agent_ids=agent, track_ids=tracked_id, window_keys=key)
            for name in names:
                arrays[name].append(values[name])
    if not arrays["residuals"]:
        raise ValueError("No matched history windows were supplied")
    result = {name: np.asarray(values) for name, values in arrays.items()}
    return result, dict(frames=frames, matched_windows=len(identities), timestamped_windows=timestamped,
                       source_group_ids=sorted(set(result["scene_ids"].tolist())))


def fit_noise_from_histories(histories, output=None, *, history_steps=None,
                             sample_fps=None, manifest_path=None, min_pool_size=2):
    """Fit an explicit training/heldout capture and optionally save its profile.

    ``histories`` is a directory (history.json + histories.pkl.gz) or a specific
    records file. A manifest is required to establish split and physical grid.
    Without per-track timestamps, history_contract must declare obs_len/fps;
    this declares uniformly sampled queues ending at the current frame.
    Existing detector/tracker provenance is carried into fit_metadata.
    """
    source = Path(histories).resolve()
    if output is not None and Path(output).exists():
        raise FileExistsError(f"Output already exists: {output}")
    directory = source if source.is_dir() else source.parent
    path = source / "histories.pkl.gz" if source.is_dir() else source
    manifest_path = Path(manifest_path).resolve() if manifest_path is not None else directory / "history.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("source_split") not in ("train", "heldout", "calibration")
            or manifest.get("evaluation_used_for_fitting") is not False or manifest.get("evaluation_only", False)):
        raise ValueError("Noise fitting requires completed train/heldout histories with evaluation_used_for_fitting=false")
    contract = manifest.get("history_contract", {})
    declared_steps = contract.get("obs_len", manifest.get("history_steps", manifest.get("obs_len")))
    declared_fps = contract.get("fps", manifest.get("sample_fps", manifest.get("fps")))
    steps = _positive_integer(history_steps if history_steps is not None else declared_steps, "history_steps")
    fps = _frequency(sample_fps if sample_fps is not None else declared_fps)
    if declared_steps is not None and steps != declared_steps:
        raise ValueError("Requested history_steps differs from the capture contract")
    if declared_fps is not None and not np.isclose(fps, float(declared_fps), rtol=0., atol=1e-12):
        raise ValueError("Requested sample_fps differs from the capture contract")
    before = file_identity(path)
    if manifest.get("histories", {}).get("sha256", before["sha256"]) != before["sha256"]:
        raise ValueError("History capture hash differs from its manifest")
    manifest_identity = file_identity(manifest_path)
    arrays, metadata = collect_history_errors(iter_history_records(path), steps, sample_fps=fps)
    if metadata["timestamped_windows"] != metadata["matched_windows"] and not {"obs_len", "fps"} <= contract.keys():
        raise ValueError("Histories without timestamps require history_contract.obs_len and history_contract.fps")
    metadata.update(source_manifest=manifest_identity, source_history_manifest=manifest_identity,
        histories=before, source_split=manifest["source_split"], evaluation_used_for_fitting=False,
        history_contract=contract)
    for name in ("runtime_pipeline", "runtime_pipeline_id", "detector_training_overlap", "detector_config", "tracker_config"):
        if name in manifest:
            metadata[name] = manifest[name]
    if file_identity(path) != before or file_identity(manifest_path) != manifest_identity:
        raise ValueError("History source changed during extraction")
    tracks = [f"{agent}|{track}" for agent, track in zip(arrays["agent_ids"], arrays["track_ids"])]
    model = fit_history_noise(arrays["residuals"], arrays["categories"], arrays["history_lengths"],
        scene_ids=arrays["scene_ids"], track_ids=tracks, keys=arrays["window_keys"],
        sample_fps=fps, min_pool_size=min_pool_size, metadata=metadata)
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", type=Path, required=True, help="Paired records file or capture directory")
    parser.add_argument("--manifest", type=Path, help="Capture manifest (default: adjacent history.json)")
    parser.add_argument("--output", type=Path, required=True, help="Fitted noise parameters JSON")
    parser.add_argument("--history-steps", type=int, help="Observed window length; defaults to capture metadata")
    parser.add_argument("--sample-fps", type=float, help="Physical sampling frequency; defaults to capture metadata")
    parser.add_argument("--min-pool-size", type=int, default=2)
    args = parser.parse_args(argv)
    model = fit_noise_from_histories(args.histories, args.output, history_steps=args.history_steps,
        sample_fps=args.sample_fps, manifest_path=args.manifest, min_pool_size=args.min_pool_size)
    print(f"Saved {model.summary()['n_pools']} noise pools to {args.output}")


if __name__ == "__main__":
    main()
