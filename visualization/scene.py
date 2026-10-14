"""Normalize current vehicle responses for rendering without changing their timelines."""

import numpy as np


def _box(value):
    if isinstance(value, dict):
        dimensions = [("length", "dx"), ("width", "dy"), ("height", "dz")]
        return np.asarray([value[axis] for axis in ("x", "y", "z")]
                          + [value.get(long, value.get(short, 0.0))
                             for long, short in dimensions]
                          + [value["yaw"]], dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)[:7].copy()


def _history(records):
    if records is None or len(records) == 0:
        return np.empty((0, 2))
    return np.asarray([[record.x, record.y] if hasattr(record, "x") else record[:2]
                       for record in records], dtype=float).reshape(-1, 2)


def _forecast(value, timeline_ms, current_ms, source_vehicle, source_object_id):
    if not value:
        return None
    means = value.get("pred")
    if isinstance(means, dict) and means:
        keys = sorted(means, key=float)
        times = np.asarray(keys, dtype=float)
        xy = np.asarray([means[key] for key in keys], dtype=float)
        covariance = value.get("cov")
        covariances = ([covariance.get(key) for key in keys]
                       if isinstance(covariance, dict) else None)
    elif value.get("t") is not None and len(value["t"]):
        times = np.asarray(value["t"], dtype=float)
        xy = np.asarray(value["xy"], dtype=float)
        covariances = value.get("cov")
    else:
        return None
    xy = xy.reshape(-1, 2)
    if len(times) != len(xy):
        raise ValueError("Visualization forecast timestamps and positions must have equal lengths.")
    cov = None
    if covariances is not None:
        cov = np.full((len(times), 2, 2), np.nan)
        for index, matrix in enumerate(covariances):
            if index >= len(times):
                break
            if matrix is not None and np.asarray(matrix).shape == (2, 2):
                cov[index] = matrix
    return {
        "timestamps": times + (timeline_ms - current_ms) / 1000.0,
        "xy": xy.copy(), "cov": cov,
        "source_vehicle": value.get("source_vehicle", source_vehicle),
        "source_object_id": value.get("source_object_id", source_object_id),
        "origin_ms": value.get("forecast_origin_ms", value.get("origin_ms", timeline_ms)),
    }


def build_scene(scenario, timestamp_ms, ego_name, responses, *, associations=None,
                received_vehicle_poses=None, iou_threshold=None):
    """Build one world-frame scene; cached peers retain their own response timestamp.

    ``associations`` is the evaluator's (matches, missed GT IDs, false indices)
    tuple. Means and covariance are copied without interpolation or extrapolation.
    """
    vehicles = {}
    for name, response in responses.items():
        if response is None:
            continue
        response_ms = int(response.get("timestamp_ms", timestamp_ms))
        tracks = [{"id": track["id"], "category": track["category"],
                   "box": _box(track["current_pos"]), "past": _history(track["tracklet"]),
                   "missed_frames": track.get("missed_frames", 0)}
                  for track in response.get("tracklets", [])]
        histories = {track["id"]: track["past"] for track in tracks}
        objects = []
        for prediction in response.get("predictions", []):
            object_id = prediction["id"]
            prediction_ms = prediction.get("timestamp")
            prediction_ms = response_ms if prediction_ms is None else int(prediction_ms)
            ego = _forecast(prediction.get("prediction"), prediction_ms,
                            timestamp_ms, name, object_id)
            fused = _forecast(prediction.get("fused_prediction"), prediction_ms,
                              timestamp_ms, name, object_id)
            shared = []
            for forecast in prediction.get("fusion_inputs", []):
                prepared = _forecast(forecast, forecast.get("origin_ms", response_ms),
                                     timestamp_ms, None, None)
                if prepared is not None:
                    shared.append(prepared)
            objects.append({
                "id": object_id, "category": prediction["category"],
                "node_type": prediction.get("node_type", 1 if ego is not None else 2),
                "box": _box(prediction["cur_location"]),
                "past": histories.get(object_id, np.empty((0, 2))),
                "ego": ego, "fused": fused, "shared": shared,
            })

        forecasts = [forecast for obj in objects
                     for forecast in (obj["ego"], obj["fused"], *obj["shared"])
                     if forecast is not None]
        # GT is sampled at the vehicle's FPS; displayed future ends at its forecasts.
        horizon_s = max((np.max(f["timestamps"]) - (response_ms - timestamp_ms) / 1000
                         for f in forecasts), default=0.0)
        fps = float(response["fps"])
        horizon_steps = max(0, int(np.floor(horizon_s * fps + 1e-6)))
        past_steps = int(response.get("observation_length",
                                     max((len(track["past"]) for track in tracks), default=1)))
        ground_truth = [{
            "id": object_id, "category": value["category"],
            "box": _box(value["current_state"]),
            "past": _history(value.get("past", [])[-past_steps:]),
            "future": _history(value.get("future", [])[:horizon_steps]),
        } for object_id, value in response.get("trajectories", {}).items()]
        for truth in ground_truth:
            truth["future_timestamps"] = ((np.arange(len(truth["future"])) + 1) / fps
                                            + (response_ms - timestamp_ms) / 1000.0)

        vehicles[name] = {
            "timestamp_ms": response_ms, "age_ms": int(timestamp_ms) - response_ms,
            "pose": dict(response.get("ego_state") or {}),
            "calibration": response.get("calibration"), "lidar_path": response.get("lidar_path"),
            "point_cloud": response.get("point_cloud"),
            "detections": [_box(value) for value in response.get("detections", [])],
            "tracks": tracks, "objects": objects, "ground_truth": ground_truth,
        }

    if associations is not None and ego_name in vehicles:
        matches, missed_ids, false_indices = associations
        ego = vehicles[ego_name]
        truths = {truth["id"]: truth for truth in ego["ground_truth"]}
        for truth_id, index in matches:
            ego["objects"][index].update(status="matched", ground_truth_id=truth_id)
            if truth_id in truths:
                truths[truth_id].update(status="matched", prediction_id=ego["objects"][index]["id"])
        for index in false_indices:
            ego["objects"][index]["status"] = "unmatched"
        for truth_id in missed_ids:
            if truth_id in truths:
                truths[truth_id]["status"] = "missed"
    ego_pose = vehicles[ego_name]["pose"]
    return {
        "scenario": scenario, "timestamp_ms": int(timestamp_ms), "ego_name": ego_name,
        "origin": np.asarray([ego_pose["x"], ego_pose["y"]], dtype=float),
        "vehicles": vehicles,
        "received_vehicle_poses": {
            name: {**record, "pose": dict(record["pose"])}
            for name, record in (received_vehicle_poses or {}).items()
        },
        "iou_threshold": iou_threshold,
    }
