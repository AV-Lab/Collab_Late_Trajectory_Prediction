"""Read canonical calibration scenes and run the checkpoint's public API.

Only observed histories, current states and scene context enter the predictor.
Complete target futures are retained separately for fitting and evaluation.
"""

from copy import deepcopy
from pathlib import Path
import pickle
import random

import numpy as np

from .bootstrap import SEED


BATCH_SIZE = 32
CONTRACT_FIELDS = (
    "obs_len", "pred_len", "sample_fps", "coordinate_frame", "coordinate_units",
    "common_coordinates", "coordinate_convention", "coordinate_transform_version",
)
COMMON_CONTRACT = {
    "coordinate_frame": "world", "coordinate_units": "meters",
    "common_coordinates": True, "coordinate_convention": "waymo_v1",
    "coordinate_transform_version": 1,
}


def _xy(values, name, allow_empty=False):
    points = np.asarray(values, dtype=np.float64)
    if allow_empty and points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError("{} must contain finite coordinate vectors with XY.".format(name))
    return points[:, :2]


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a nonempty string.".format(name))
    return value


def load_dataset(path):
    """Load a trusted local schema-3 archive without opening sensor files."""
    with Path(path).open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, dict) or not isinstance(dataset.get("metadata"), dict):
        raise ValueError("Calibration archive requires metadata and samples.")
    metadata = dataset["metadata"]
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 3 or metadata.get("layout") != "scene":
        raise ValueError("Calibration requires schema_version=3, layout='scene'.")
    contract = metadata.get("input_contract")
    if (not isinstance(contract, dict) or type(contract.get("schema_version")) is not int
            or contract["schema_version"] != 1):
        raise ValueError("Archive requires input_contract schema_version=1.")
    for name in CONTRACT_FIELDS:
        if (name not in metadata or metadata[name] != contract.get(name)
                or (name != "sample_fps" and type(metadata[name]) is not type(contract.get(name)))):
            raise ValueError("metadata.{} must match input_contract.".format(name))
    for name in ("obs_len", "pred_len"):
        if type(metadata[name]) is not int or metadata[name] < (2 if name == "obs_len" else 1):
            raise ValueError("metadata.{} has an invalid length.".format(name))
    fps = metadata["sample_fps"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not np.isfinite(fps) or fps <= 0:
        raise ValueError("metadata.sample_fps must be finite and positive.")
    for name, expected in COMMON_CONTRACT.items():
        if type(metadata[name]) is not type(expected) or metadata[name] != expected:
            raise ValueError("metadata.{} must be {!r}.".format(name, expected))
    samples = dataset.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Archive samples must be a nonempty list.")
    seen, scenario_groups = set(), {}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("Scene {} must be a dictionary.".format(index))
        sample_id = _identifier(sample.get("sample_id"), "sample_id")
        if sample_id in seen:
            raise ValueError("Duplicate scene sample_id: {}".format(sample_id))
        seen.add(sample_id)
        scenario = _identifier(sample.get("scenario_id"), "scenario_id")
        group = _identifier(sample.get("source_group_id"), "source_group_id")
        if scenario_groups.setdefault(scenario, group) != group:
            raise ValueError("One scenario has conflicting physical source groups.")
        scene_metadata = sample.get("metadata")
        if scene_metadata is not None:
            if not isinstance(scene_metadata, dict):
                raise ValueError("Scene metadata must be a dictionary or None.")
            for field in ("sample_fps", "coordinate_convention", "coordinate_transform_version"):
                if field in scene_metadata and scene_metadata[field] != metadata[field]:
                    raise ValueError("Scene {} conflicts with the archive contract.".format(field))
        if not isinstance(sample.get("objects"), list):
            raise ValueError("Scene objects must be a list.")
        for obj in sample["objects"]:
            if not isinstance(obj, dict):
                raise ValueError("Scene objects must be dictionaries.")
            _identifier(obj.get("agent_class"), "agent_class")
            if obj.get("track_id") is None:
                raise ValueError("Object track_id is required.")
            if any(not isinstance(obj.get(role), (bool, np.bool_))
                   for role in ("target_agent", "use_as_past", "use_as_context")):
                raise ValueError("Objects require boolean target/history/context roles.")
            if not isinstance(obj.get("current_state"), dict):
                raise ValueError("Object current_state is required.")
            _xy([obj["current_state"].get("center")], "current_state.center")
            past = _xy(obj.get("past_traj"), "past_traj", allow_empty=True)
            if len(past) >= metadata["obs_len"]:
                raise ValueError("past_traj excludes the current point and must be shorter than obs_len.")
            future = obj.get("future_traj")
            if future is not None:
                _xy(future, "future_traj", allow_empty=True)
    calibration_rows(dataset)
    return dataset


def calibration_rows(dataset):
    """Index complete scoring targets while leaving their scene context intact."""
    metadata = dataset["metadata"]
    minimum = metadata.get("required_obs_len", metadata["obs_len"])
    if type(minimum) is not int or not 2 <= minimum <= metadata["obs_len"]:
        raise ValueError("required_obs_len must be between two and obs_len.")
    rows = []
    for scene_index, sample in enumerate(dataset["samples"]):
        for object_index, obj in enumerate(sample["objects"]):
            if not obj["target_agent"] or not obj["use_as_past"]:
                continue
            past = _xy(obj["past_traj"], "past_traj", allow_empty=True)
            current = _xy([obj["current_state"]["center"]], "current_state.center")
            raw_future = obj.get("future_traj")
            future = _xy([] if raw_future is None else raw_future, "future_traj", allow_empty=True)
            if len(past) + 1 < minimum or len(future) != metadata["pred_len"]:
                continue
            rows.append({
                "scene_index": scene_index, "object_index": object_index,
                "track_id": "calibration-{}-{}".format(scene_index, object_index),
                "category": obj["agent_class"], "history": np.concatenate((past, current)),
                "future": future, "source_group_id": sample["source_group_id"],
            })
    if not rows:
        raise ValueError("Archive has no targets with sufficient history and complete futures.")
    return rows


def create_predictor(checkpoint_path):
    """Restore the model's preprocessing, weights and public prediction modes."""
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
    """Check the saved input contract rather than inferring it from directory names."""
    config = predictor.config
    saved = config.get("data", {}).get("input_contract")
    if (not isinstance(saved, dict) or type(saved.get("schema_version")) is not int
            or saved["schema_version"] != 1):
        raise ValueError("Checkpoint must contain data.input_contract schema_version=1.")
    layout = config.get("dataloader", {}).get("name")
    if layout not in ("track", "scene"):
        raise ValueError("Checkpoint must declare track or scene layout.")
    for name in CONTRACT_FIELDS:
        actual, expected = saved.get(name), metadata.get(name)
        matches = (isinstance(actual, (int, float)) and not isinstance(actual, bool)
                   and np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-9, atol=0)
                   if name == "sample_fps" else type(actual) is type(expected) and actual == expected)
        if not matches:
            raise ValueError("Checkpoint input_contract.{} does not match the archive.".format(name))
    for name in ("obs_len", "pred_len"):
        value = config["model"]["params"].get(name)
        if type(value) is not int or value != saved[name]:
            raise ValueError("Checkpoint model {} conflicts with its saved input contract.".format(name))
    for name, expected in COMMON_CONTRACT.items():
        if type(saved.get(name)) is not type(expected) or saved[name] != expected:
            raise ValueError("Checkpoint {} must be {!r}.".format(name, expected))
    if config.get("map_preprocessor") is not None or config.get("map_encoder") is not None:
        if not metadata.get("capabilities", {}).get("maps", {}).get("all_scenes", False):
            raise ValueError("Map-dependent checkpoint requires map data in every archive scene.")
    return {
        **{name: saved[name] for name in CONTRACT_FIELDS}, "layout": layout,
        "coordinate_source_profile": saved.get("coordinate_source_profile"),
        "timing_check": "checkpoint_verified",
        "context_policy": "preserved_scene_context" if layout == "scene" else "independent_tracks",
        "history_policy": "past_traj_plus_current_state_center",
        "alignment_policy": "native_forecast_timestamps",
        "mode_selection": "runtime_output_index_0",
    }


def _input_object(obj, scene_index, object_index):
    """Copy observed fields only; synthetic IDs prevent collisions across scenes."""
    state = deepcopy(obj["current_state"])
    state["center"] = _xy([state["center"]], "center")[0].tolist()
    return {
        **{name: deepcopy(obj[name]) for name in ("past_heading", "past_velocity", "metadata")
           if name in obj},
        "track_id": "calibration-{}-{}".format(scene_index, object_index),
        "agent_class": obj["agent_class"],
        "target_agent": bool(obj["target_agent"]),
        "use_as_past": bool(obj["use_as_past"]),
        "use_as_context": bool(obj["use_as_context"]),
        "past_traj": _xy(obj["past_traj"], "past_traj", allow_empty=True).tolist(),
        "current_state": state, "future_traj": None,
    }


def prediction_batches(dataset, layout, batch_size=BATCH_SIZE):
    """Yield public-API batches and corresponding scoring row indices."""
    if layout not in ("track", "scene"):
        raise ValueError("Prediction layout must be track or scene.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    rows = calibration_rows(dataset)
    samples = dataset["samples"]
    if layout == "track":
        for start in range(0, len(rows), batch_size):
            indices = list(range(start, min(start + batch_size, len(rows))))
            objects = [_input_object(samples[rows[i]["scene_index"]]["objects"][rows[i]["object_index"]],
                                     rows[i]["scene_index"], rows[i]["object_index"]) for i in indices]
            yield {"layout": "track", "objects": objects}, indices
        return
    by_scene = {}
    for row_index, row in enumerate(rows):
        by_scene.setdefault(row["scene_index"], []).append(row_index)
    scene_indices = list(by_scene)
    for start in range(0, len(scene_indices), batch_size):
        batch_indices = scene_indices[start:start + batch_size]
        scenes, objects, selected = [], [], []
        for scene_index in batch_indices:
            raw = samples[scene_index]
            scene = {key: deepcopy(raw[key]) for key in (
                "sample_id", "scenario_id", "scene_id", "source_group_id", "metadata",
                "map_data", "sensors_data", "ego") if key in raw}
            scene["objects"] = [_input_object(obj, scene_index, index)
                                for index, obj in enumerate(raw["objects"])]
            include = [index for index, obj in enumerate(scene["objects"])
                       if (obj["use_as_past"] or obj["use_as_context"])
                       and len(obj["past_traj"]) + 1 >= 2]
            scenes.append(scene)
            objects.append([scene["objects"][index] for index in include])
            selected.append(include)
        yield {
            "layout": "scene", "samples": scenes, "objects": objects,
            "selected_indices": selected,
            "num_objects": [len(scene["objects"]) for scene in scenes],
            "num_agents": [len(scene) for scene in objects],
            "sample_ids": [scene["sample_id"] for scene in scenes],
            "scenario_ids": [scene["scenario_id"] for scene in scenes],
            "scene_ids": [scene.get("scene_id") for scene in scenes],
        }, [index for scene_index in batch_indices for index in by_scene[scene_index]]


def collect_predictions(predictor, dataset, layout, batch_size=BATCH_SIZE):
    """Select mode zero and recover scoring targets by their stable IDs."""
    rows = calibration_rows(dataset)
    horizon = dataset["metadata"]["pred_len"]
    means = np.empty((len(rows), horizon, 2), dtype=np.float64)
    covariances = np.empty((len(rows), horizon, 2, 2), dtype=np.float64)
    for batch, indices in prediction_batches(dataset, layout, batch_size):
        output = predictor.predict(batch)
        trajectories = np.asarray(output["trajectories"])
        covariance = np.asarray(output.get("covariances"))
        track_ids = list(output["track_ids"])
        if (trajectories.ndim != 4 or trajectories.shape[1] < 1
                or trajectories.shape[2:] != (horizon, 2)
                or len(track_ids) != trajectories.shape[0]):
            raise ValueError("Prediction trajectories must have shape [N,K,pred_len,2].")
        if covariance.shape != trajectories.shape + (2,):
            raise ValueError("Certificate estimation requires covariance [N,K,pred_len,2,2].")
        if len(set(track_ids)) != len(track_ids):
            raise ValueError("Predictor returned duplicate track IDs.")
        positions = {track_id: index for index, track_id in enumerate(track_ids)}
        expected = [rows[index]["track_id"] for index in indices]
        if any(track_id not in positions for track_id in expected):
            raise ValueError("Predictor did not return every calibration target.")
        order = [positions[track_id] for track_id in expected]
        means[indices], covariances[indices] = trajectories[order, 0], covariance[order, 0]
        print("Predicted {}/{} calibration trajectories".format(max(indices) + 1, len(rows)), flush=True)
    return means, covariances
