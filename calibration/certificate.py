"""Create a bootstrap certificate from checkpoint inference or a frozen experiment.

Fresh inference (with TrajZoo on PYTHONPATH):
    python calibration/certificate.py --checkpoint best.pth \
        --dataset trajectories.pkl --output outputs/calibration/certificate.json
Frozen export, copying fitted values without predictor inference or refitting:
    python calibration/certificate.py --from-experiment fitted_profiles.json \
        --output outputs/calibration/certificate.json

This module owns inference, provenance and file handling. Bootstrap estimation
and validation live in bounds.py. Confidence is approximate per group, and
transfer to deployment inputs is assumed rather than verified.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import tempfile

import numpy as np

if __package__:
    from .bounds import (
        BOOTSTRAP_DRAWS, CONFIDENCE, MIN_VALID_ROWS, SEED,
        _validate_frozen_bin, estimate_bootstrap_profiles,
    )
else:
    from bounds import (
        BOOTSTRAP_DRAWS, CONFIDENCE, MIN_VALID_ROWS, SEED,
        _validate_frozen_bin, estimate_bootstrap_profiles,
    )


BATCH_SIZE = 32


def load_dataset(path):
    """Validate the standalone schema without importing TrajZoo or source data."""
    with Path(path).open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("metadata"), dict):
        raise ValueError("PKL must contain metadata and samples dictionaries/list.")
    metadata = dataset["metadata"]
    if metadata.get("schema_version") != 1:
        raise ValueError("Only calibration PKL schema_version=1 is supported.")
    for name in ("obs_len", "pred_len"):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"metadata.{name} must be a positive integer.")
    fps = metadata.get("sample_fps")
    if (isinstance(fps, bool) or not isinstance(fps, (int, float))
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("metadata.sample_fps must be finite and positive.")
    for name in ("coordinate_frame", "coordinate_units"):
        if not isinstance(metadata.get(name), str) or not metadata[name].strip():
            raise ValueError(f"metadata.{name} must be a nonempty string.")

    samples = dataset.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("PKL samples must be a nonempty list.")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} must be a dictionary.")
        category = sample.get("category")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"Sample {index} requires a category.")
        for name, length in (("past", metadata["obs_len"]), ("future", metadata["pred_len"])):
            values = sample.get(name)
            if (not isinstance(values, np.ndarray) or values.dtype != np.float32
                    or values.shape != (length, 2) or not np.isfinite(values).all()):
                raise ValueError(
                    f"Sample {index} {name} must be a finite float32 array [{length},2]."
                )
    return dataset


def create_predictor(checkpoint_path):
    """Restore TrajZoo's predictor, retaining its own preprocessing and modes."""
    import torch
    from predictor import Predictor

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    return Predictor.from_checkpoint(checkpoint_path=str(checkpoint_path), device=device)


def validate_predictor(predictor, metadata):
    """Check the checkpoint's layout, lengths and available timing declarations."""
    config = predictor.config
    model_parameters = config["model"]["params"]
    layout = config.get("dataloader", {}).get("name")
    if layout not in ("track", "scene"):
        raise ValueError("Checkpoint must declare a track or scene dataloader layout.")
    if config.get("map_preprocessor") is not None or config.get("map_encoder") is not None:
        raise ValueError("The XY-only calibration PKL cannot supply map-dependent inputs.")
    for name in ("obs_len", "pred_len"):
        if model_parameters.get(name) != metadata[name]:
            raise ValueError(f"Checkpoint {name} does not match the calibration PKL.")

    declared_fps = []
    for source in (config, model_parameters):
        for name in ("sample_fps", "fps"):
            if source.get(name) is not None:
                declared_fps.append(float(source[name]))
    dataset_configs = config.get("datasets", [])
    if isinstance(dataset_configs, dict):
        dataset_configs = dataset_configs.values()
    for dataset_config in dataset_configs:
        parameters = dataset_config.get("params", {})
        value = parameters.get("sample_fps")
        if value is None:
            value = parameters.get("native_fps")
        if value is not None:
            declared_fps.append(float(value))
    if model_parameters.get("dt") is not None:
        dt = float(model_parameters["dt"])
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Checkpoint dt must be finite and positive.")
        declared_fps.append(1.0 / dt)
    for fps in declared_fps:
        if not np.isfinite(fps) or not np.isclose(fps, metadata["sample_fps"], rtol=1e-9, atol=0):
            raise ValueError("Checkpoint sampling frequency does not match the calibration PKL.")

    return {
        "layout": layout,
        "obs_len": metadata["obs_len"],
        "pred_len": metadata["pred_len"],
        "sample_fps": metadata["sample_fps"],
        "timing_check": "checkpoint_verified" if declared_fps else "assumed_from_dataset",
        "context_policy": "singleton_scene" if layout == "scene" else "independent_tracks",
        "history_policy": "raw_complete_dataset_coordinates",
        "alignment_policy": "native_forecast_timestamps",
        "coordinate_frame": metadata["coordinate_frame"],
        "coordinate_units": metadata["coordinate_units"],
    }


def prediction_batches(dataset, layout, batch_size=BATCH_SIZE):
    """Yield batches with stable row IDs and no ground-truth futures as inputs."""
    if layout not in ("track", "scene"):
        raise ValueError("Prediction layout must be track or scene.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    samples = dataset["samples"]
    for start in range(0, len(samples), batch_size):
        indices = list(range(start, min(start + batch_size, len(samples))))
        objects = []
        for index in indices:
            sample = samples[index]
            objects.append({
                "track_id": f"calibration-{index}",
                "agent_class": sample["category"],
                "target_agent": True,
                "use_as_past": True,
                "use_as_context": True,
                "past_traj": sample["past"].tolist(),
                "current_state": None,  # past already contains the current position
                "future_traj": None,
            })
        if layout == "track":
            yield {"layout": "track", "objects": objects}, indices
            continue

        # Each row is its own scene; a batch does not imply shared scene context.
        scene_ids = [f"calibration-row-{index}" for index in indices]
        scenes = [
            {"sample_id": scene_id, "scenario_id": None, "scene_id": None, "objects": [obj]}
            for scene_id, obj in zip(scene_ids, objects)
        ]
        yield {
            "layout": "scene",
            "samples": scenes,
            "objects": [[obj] for obj in objects],
            "selected_indices": [[0] for _ in objects],
            "num_objects": [1] * len(objects),
            "num_agents": [1] * len(objects),
            "sample_ids": scene_ids,
            "scenario_ids": [None] * len(objects),
            "scene_ids": [None] * len(objects),
        }, indices


def collect_predictions(predictor, dataset, layout, batch_size=BATCH_SIZE):
    """Select mode zero exactly as COLTP does and restore original sample order."""
    count = len(dataset["samples"])
    horizon = dataset["metadata"]["pred_len"]
    means = np.empty((count, horizon, 2), dtype=np.float64)
    covariances = np.empty((count, horizon, 2, 2), dtype=np.float64)
    for batch, indices in prediction_batches(dataset, layout, batch_size):
        prediction = predictor.predict(batch)
        trajectories = np.asarray(prediction["trajectories"])
        if prediction.get("covariances") is None:
            raise ValueError("Certificate estimation requires reported covariance from the predictor.")
        covariance = np.asarray(prediction["covariances"])
        if (trajectories.ndim != 4 or trajectories.shape[0] != len(indices)
                or trajectories.shape[1] < 1 or trajectories.shape[2:] != (horizon, 2)):
            raise ValueError("Prediction trajectories must have shape [N,K,pred_len,2].")
        if covariance.shape != trajectories.shape + (2,):
            raise ValueError("Prediction covariance must have shape [N,K,pred_len,2,2].")
        track_ids = list(prediction["track_ids"])
        expected_ids = [f"calibration-{index}" for index in indices]
        if len(track_ids) != len(indices) or set(track_ids) != set(expected_ids):
            raise ValueError("Predictor must return each calibration track ID exactly once.")
        output_index = {track_id: index for index, track_id in enumerate(track_ids)}
        order = [output_index[track_id] for track_id in expected_ids]
        means[indices] = trajectories[order, 0]
        covariances[indices] = covariance[order, 0]
        print(f"Predicted {indices[-1] + 1}/{count} calibration trajectories", flush=True)
    return means, covariances


def _file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def _save_certificate(certificate, output_path):
    """Publish complete JSON without replacing a previous certificate."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.") as handle:
        json.dump(certificate, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.link(handle.name, path)


def _settings():
    return {
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": SEED,
        "min_fit_rows": MIN_VALID_ROWS,
        "confidence": CONFIDENCE,
        "uncertainty_bins": "fit-only covariance-trace quintiles; duplicate edges removed",
        "resampling": "whole trajectories within each category; one draw shared across horizons",
        "moment": "uncentered raw error second moment",
        "radius": "95th percentile bootstrap spectral-norm deviation",
        "confidence_scope": "approximate 95% per group; not simultaneous",
    }


def _assumptions():
    return {
        "sample_independence_verified": False,
        "source_group_ids_available": False,
        "population": "exported calibration sample distribution within category and trace bin",
        "bootstrap": "empirical trajectory resampling; finite-sample confidence is not established",
        "conditional_moments": "a group moment is an approximation for inputs assigned to that group",
        "transfer_to_runtime": "group second-moment transfer is assumed, not verified",
        "cross_source_error_moments": "zero cross-error moments are required by the gate, not established here",
        "input_scope": "complete histories; no neighbouring agents, maps or interpolated forecast points",
        "confidence_scope": "approximate per group; no simultaneous or deployment guarantee",
    }


def _new_certificate(checkpoint, dataset, contract, profiles, settings, provenance):
    usable = sum(any(item["usable_for_gate"] for item in row["bins"]) for row in profiles)
    return {
        "schema_version": 2,
        "estimator": "bootstrap_raw_moment",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "bootstrap_group_moment_approximate" if usable else "no_usable_profiles",
        "usable_profile_count": usable,
        "checkpoint": checkpoint,
        "dataset": dataset,
        "inference": contract,
        "assumptions": _assumptions(),
        "settings": settings,
        "fit_provenance": provenance,
        "profiles": profiles,
    }


def create_certificate(checkpoint_path, dataset_path, output_path):
    """Run new predictor inference and fit on all supplied calibration samples."""
    if Path(output_path).exists() or Path(output_path).is_symlink():
        raise FileExistsError(f"Certificate already exists: {output_path}")
    dataset = load_dataset(dataset_path)
    checkpoint_identity = _file_identity(checkpoint_path)
    dataset_identity = _file_identity(dataset_path)
    predictor = create_predictor(checkpoint_path)
    contract = validate_predictor(predictor, dataset["metadata"])
    means, covariances = collect_predictions(predictor, dataset, contract["layout"])
    targets = np.stack([sample["future"] for sample in dataset["samples"]])
    categories = [sample["category"] for sample in dataset["samples"]]
    profiles, exclusions = estimate_bootstrap_profiles(
        means, targets, covariances, categories, contract["sample_fps"])
    result = _new_certificate(
        {**checkpoint_identity, "model_name": predictor.config["model"]["name"]},
        {**dataset_identity, "metadata": dataset["metadata"], "n_samples": len(categories)},
        {**contract, "seed": SEED, "batch_size": BATCH_SIZE,
         "device": str(predictor.device), "mode_selection": "runtime_output_index_0",
         "checkpoint_config": predictor.config},
        profiles, _settings(),
        {"selection": "all supplied calibration trajectories", "n_fit": len(categories),
         "heldout_used_for_fitting": False})
    result["exclusions"] = exclusions
    _save_certificate(result, output_path)
    print(f"Certificate saved to: {Path(output_path).resolve()}", flush=True)
    return result


def export_experiment(experiment_path, output_path):
    """Wrap frozen vehicle-only fitted values; never rerun or refit the experiment."""
    if Path(output_path).exists() or Path(output_path).is_symlink():
        raise FileExistsError(f"Certificate already exists: {output_path}")
    path = Path(experiment_path)
    artifact = json.loads(path.read_text())
    if artifact.get("format") != "coltp_bootstrap_moment_experiment_v1":
        raise ValueError("Unsupported frozen bootstrap experiment format.")
    split_path = path.with_name("vehicle_split.json")
    split = json.loads(split_path.read_text())
    fit, heldout = split["fit_indices"], split["heldout_indices"]
    if (not fit or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in fit + heldout)
            or len(set(fit)) != len(fit) or len(set(heldout)) != len(heldout)
            or set(fit) & set(heldout)):
        raise ValueError("Frozen experiment requires disjoint unique fitting and heldout IDs.")
    fps = float(artifact["inference"]["sample_fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Frozen experiment requires a positive sample_fps.")
    by_horizon = {}
    for item in artifact["profiles"]:
        if item["category"] != "vehicle":
            raise ValueError("This frozen experiment export requires vehicle-only profiles.")
        by_horizon.setdefault(item["native_horizon_ms"], []).append(item)
    cutpoints = artifact["uncertainty_cutpoints"]
    if {str(h) for h in by_horizon} != set(cutpoints):
        raise ValueError("Frozen horizons and uncertainty cutpoints must agree.")
    profiles = []
    for horizon, items in sorted(by_horizon.items()):
        step = int(round(horizon * fps / 1000))
        if step < 1 or not np.isclose(step / fps, horizon / 1000):
            raise ValueError("Frozen horizon does not belong to the declared forecast grid.")
        edges = cutpoints[str(horizon)]
        edge_array = np.asarray(edges, dtype=float)
        if (edge_array.ndim != 1 or not np.isfinite(edge_array).all()
                or np.any(edge_array < 0) or np.any(np.diff(edge_array) <= 0)):
            raise ValueError("Frozen uncertainty cutpoints must be finite, increasing and nonnegative.")
        ordered = sorted(items, key=lambda item: item["uncertainty_bin"])
        if [item["uncertainty_bin"] for item in ordered] != list(range(len(edges) + 1)):
            raise ValueError("Frozen bins must form a complete ordered partition.")
        bins = [_validate_frozen_bin(item) for item in ordered]
        if sum(item["n_valid"] for item in bins) > len(fit):
            raise ValueError("Frozen bin counts exceed the recorded fitting population.")
        profiles.append({"category": "vehicle", "horizon_step": step,
                         "horizon_seconds": horizon / 1000,
                         "uncertainty_cutpoints": edges, "bins": bins})
    dataset = deepcopy(artifact["dataset"])
    dataset["n_samples"] = len(fit)
    dataset["selection"] = "frozen vehicle fitting subset; see fit_provenance"
    result = _new_certificate(
        artifact["checkpoint"], dataset, artifact["inference"], profiles, artifact["settings"],
        {"selection": "frozen external vehicle fitting subset",
         "n_fit": len(fit), "n_heldout": len(heldout), "fit_indices": fit,
         "heldout_used_for_fitting": False, "experiment": _file_identity(path),
         "vehicle_split": _file_identity(split_path),
         "source_split": split.get("source_split"),
         "fitted_values": "copied exactly; no prediction, fitting or radius recomputation"})
    _save_certificate(result, output_path)
    print(f"Exported {len(fit)} frozen fitting trajectories to: {Path(output_path).resolve()}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--from-experiment", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.from_experiment is not None:
        if args.checkpoint is not None or args.dataset is not None:
            parser.error("--from-experiment cannot be combined with --checkpoint or --dataset")
        export_experiment(args.from_experiment, args.output)
    else:
        if args.checkpoint is None or args.dataset is None:
            parser.error("supply --checkpoint and --dataset, or --from-experiment")
        create_certificate(args.checkpoint, args.dataset, args.output)


if __name__ == "__main__":
    main()
