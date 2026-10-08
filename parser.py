import hashlib
import json
import math
import os
import numpy as np
import yaml

SUPPORTED_VEHICLE_TYPES = {"basic", "aggregating", "broadcasting", "hybrid"}
SUPPORTED_DETECTORS = {"gt", "gt_occ", "centerpoint"}
SUPPORTED_PREDICTOR_LAYOUTS = {"track", "scene"}
SUPPORTED_TRACKERS = {"id_association", "metric_association"}
SUPPORTED_FUSION_TYPES = {"gp", "gp_vector", "linear_fusion"}
SPLITS = {"train", "valid", "test"}


def _config_block(value, path, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{path}: must be a dictionary")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"{path}: unsupported settings {sorted(unknown)}")
    return value


def _config_number(value, path, minimum=0, inclusive=False, integer=False):
    valid_type = isinstance(value, int) if integer else isinstance(value, (int, float))
    if (not valid_type or isinstance(value, bool) or not math.isfinite(value)
            or (value < minimum if inclusive else value <= minimum)):
        kind = "integer" if integer else "number"
        comparison = "at least" if inclusive else "greater than"
        raise ValueError(f"{path}: must be a finite {kind} {comparison} {minimum}")


def _config_bool(value, path):
    if not isinstance(value, bool):
        raise ValueError(f"{path}: must be a boolean")


def _validate_categories(vehicle_key, vehicle):
    """Validate each category's alignment, gate and fusion independently."""
    prefix = f"vehicles.{vehicle_key}"
    for name in ("alignment", "gates", "fusion"):
        if name in vehicle:
            raise ValueError(f"{prefix}.{name}: move settings into category_l and category_s")
    gate_defaults = {
        "kf": {
            "mode": "simple", "min_streak": 2, "cov_ratio_thr": 2.0, "nis_thr": 5.99,
            "w_cov": 0.5, "w_miss": 0.3, "w_nis": 0.2,
            "kfri_thr": 0.6, "cov_cap": 2.0, "nis_cap": 2.0,
        },
        "consensus": {"threshold": 2.0, "min_predictions": 3, "min_overlap": 0.5},
    }
    for category in ("category_l", "category_s"):
        path = f"{prefix}.{category}"
        if category not in vehicle:
            raise ValueError(f"{path}: required")
        branch = _config_block(vehicle[category], path, {"alignment", "gate", "fusion"})

        alignment = branch.setdefault("alignment", {})
        _config_block(alignment, path + ".alignment", {"align", "min_points"})
        alignment.setdefault("align", False)
        alignment.setdefault("min_points", 10)
        _config_bool(alignment["align"], path + ".alignment.align")
        _config_number(alignment["min_points"], path + ".alignment.min_points", integer=True)

        gate = branch.setdefault("gate", {"type": "none"})
        gate_path = path + ".gate"
        _config_block(gate, gate_path, {"type", *gate_defaults["kf"], *gate_defaults["consensus"]})
        kind = gate.get("type", "none")
        supported = {"none", "kf"} if category == "category_l" else {"none", "consensus"}
        if not isinstance(kind, str) or kind not in supported:
            raise ValueError(f"{gate_path}.type: must be one of {sorted(supported)}")
        defaults = gate_defaults.get(kind, {})
        _config_block(gate, gate_path, {"type", *defaults})
        gate.setdefault("type", "none")
        for name, default in defaults.items():
            gate.setdefault(name, default)
        if kind == "kf":
            if gate["mode"] not in ("simple", "kfri"):
                raise ValueError(f"{gate_path}.mode: must be simple or kfri")
            _config_number(gate["min_streak"], gate_path + ".min_streak", integer=True, inclusive=True)
            for name in ("cov_ratio_thr", "nis_thr", "cov_cap", "nis_cap"):
                _config_number(gate[name], gate_path + "." + name)
            for name in ("w_cov", "w_miss", "w_nis", "kfri_thr"):
                _config_number(gate[name], gate_path + "." + name, inclusive=True)
        elif kind == "consensus":
            _config_number(gate["threshold"], gate_path + ".threshold")
            _config_number(gate["min_overlap"], gate_path + ".min_overlap", inclusive=True)
            _config_number(gate["min_predictions"], gate_path + ".min_predictions",
                           minimum=3, inclusive=True, integer=True)

        fusion_path = path + ".fusion"
        if "fusion" not in branch:
            raise ValueError(f"{fusion_path}: required")
        fusion = _config_block(branch["fusion"], fusion_path, {"type", "workers"})
        kind = fusion.get("type")
        supported = SUPPORTED_FUSION_TYPES if category == "category_l" else {"gp", "gp_vector"}
        if not isinstance(kind, str) or kind not in supported:
            raise ValueError(f"{fusion_path}.type: must be one of {sorted(supported)}")
        allowed = {"type"}
        if kind == "gp_vector":
            allowed.add("workers")
            fusion.setdefault("workers", 1)
            _config_number(fusion["workers"], fusion_path + ".workers", integer=True)
        _config_block(fusion, fusion_path, allowed)


def _bootstrap_profile(profile, path):
    """Validate and index one horizon's frozen uncertainty bins."""
    cutpoints = profile.get("uncertainty_cutpoints")
    if not isinstance(cutpoints, list):
        raise ValueError(f"{path}.uncertainty_cutpoints: must be a list")
    for index, value in enumerate(cutpoints):
        _config_number(value, f"{path}.uncertainty_cutpoints[{index}]", inclusive=True)
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
        _config_number(bin_index, location + ".uncertainty_bin", integer=True, inclusive=True)
        if bin_index != index:
            raise ValueError(f"{location}.uncertainty_bin: bins must be complete and ordered")
        _config_number(entry.get("n_valid"), location + ".n_valid", integer=True, inclusive=True)
        usable = entry.get("usable_for_gate")
        _config_bool(usable, location + ".usable_for_gate")
        if not usable:
            if entry.get("M_hat") is not None or entry.get("radius") is not None:
                raise ValueError(f"{location}: unavailable bins must have null moment and radius")
            indexed.append(None)
            continue
        if entry["n_valid"] == 0:
            raise ValueError(f"{location}.n_valid: usable bins require observations")
        _config_number(entry.get("radius"), location + ".radius", inclusive=True)
        raw = entry.get("M_hat")
        if (not isinstance(raw, list) or len(raw) != 2
                or any(not isinstance(row, list) or len(row) != 2 for row in raw)
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for row in raw for v in row)):
            raise ValueError(f"{location}.M_hat: must be a finite numeric 2x2 matrix")
        moment = np.asarray(raw, dtype=np.float64)
        tolerance = 1e-12 + 1e-10 * np.max(np.abs(moment))
        if np.max(np.abs(moment - moment.T)) > tolerance:
            raise ValueError(f"{location}.M_hat: must be symmetric")
        moment = 0.5 * moment + 0.5 * moment.T
        if np.linalg.eigvalsh(moment)[0] < -tolerance:
            raise ValueError(f"{location}.M_hat: must be positive semidefinite")
        indexed.append({"M_hat": moment.tolist(), "radius": float(entry["radius"])})
    return {"method": "bootstrap_raw_moment", "cutpoints": cutpoints, "bins": indexed}


def _certificate_profiles(certificate, path, inference):
    """Index usable native-point bounds; missing profiles remain unavailable."""
    if type(certificate.get("schema_version")) is not int or certificate["schema_version"] != 2:
        raise ValueError(f"{path}: requires bootstrap certificate schema_version=2; regenerate legacy certificates")
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
        _config_number(step, location + ".horizon_step", integer=True)
        if step > inference["pred_len"]:
            raise ValueError(f"{location}.horizon_step: exceeds prediction length")
        seconds = profile.get("horizon_seconds")
        _config_number(seconds, location + ".horizon_seconds")
        if not math.isclose(seconds, step / inference["sample_fps"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{location}.horizon_seconds: inconsistent with horizon_step and sample_fps")
        horizon_ms = round(seconds * 1000)
        if horizon_ms <= 0 or (category, horizon_ms) in seen:
            raise ValueError(f"{location}: duplicate or invalid category/horizon in milliseconds")
        seen.add((category, horizon_ms))
        bounds = _bootstrap_profile(profile, location)
        if any(value is not None for value in bounds["bins"]):
            indexed.setdefault(category, {})[horizon_ms] = bounds
    if not indexed:
        raise ValueError(f"{path}: certificate has no usable profiles")
    usable_count = sum(len(entries) for entries in indexed.values())
    if "usable_profile_count" in certificate and certificate["usable_profile_count"] != usable_count:
        raise ValueError(f"{path}.usable_profile_count: inconsistent with profiles")
    return indexed


def _load_calibration_by_vehicle(vehicles, fps, ego_vehicle):
    """Read certificates once, verify predictor identity, and keep them separate."""
    if not any(vehicle["type"] == "aggregating" for vehicle in vehicles.values()):
        return {}
    certificate_cache, checkpoint_hashes, registry = {}, {}, {}
    for vehicle_id, vehicle in vehicles.items():
        if vehicle["type"] == "basic" and vehicle_id != ego_vehicle:
            continue
        predictor = vehicle["predictor"]
        path = f"vehicles.{vehicle_id}.predictor.calibration_certificate"
        certificate_path = predictor.get("calibration_certificate")
        if not isinstance(certificate_path, str) or not certificate_path.strip():
            raise ValueError(f"{path}: required for fusion")
        checkpoint_path = predictor.get("checkpoint")
        if not checkpoint_path:
            raise ValueError(f"{path}: requires predictor.checkpoint to verify certificate identity")
        certificate_path = os.path.realpath(os.path.expanduser(certificate_path))
        checkpoint_path = os.path.realpath(os.path.expanduser(checkpoint_path))
        try:
            if certificate_path not in certificate_cache:
                with open(certificate_path, "r") as handle:
                    certificate_cache[certificate_path] = json.load(handle)
            if checkpoint_path not in checkpoint_hashes:
                digest = hashlib.sha256()
                with open(checkpoint_path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                checkpoint_hashes[checkpoint_path] = digest.hexdigest()
        except (OSError, ValueError) as error:
            raise ValueError(f"{path}: cannot load certificate or checkpoint: {error}") from error
        certificate = certificate_cache[certificate_path]
        if (not isinstance(certificate, dict) or type(certificate.get("schema_version")) is not int
                or certificate["schema_version"] != 2):
            raise ValueError(f"{path}: requires bootstrap certificate schema_version=2; regenerate legacy certificates")
        identity = certificate.get("checkpoint")
        if not isinstance(identity, dict) or identity.get("sha256") != checkpoint_hashes[checkpoint_path]:
            raise ValueError(f"{path}: checkpoint SHA256 does not match predictor.checkpoint")
        inference = certificate.get("inference")
        if not isinstance(inference, dict):
            raise ValueError(f"{path}.inference: must be a dictionary")
        expected = {
            "layout": predictor["layout"],
            "obs_len": round(vehicle["parameters"]["observed_past_s"] * fps),
            "pred_len": round(vehicle["parameters"]["prediction_horizon_s"] * fps),
            "mode_selection": "runtime_output_index_0",
            "alignment_policy": "native_forecast_timestamps",
            "coordinate_frame": "world", "coordinate_units": "meters",
        }
        for name in ("obs_len", "pred_len"):
            _config_number(inference.get(name), path + ".inference." + name, integer=True)
        for name, value in expected.items():
            if inference.get(name) != value:
                raise ValueError(f"{path}.inference.{name}: expected {value!r}")
        _config_number(inference.get("sample_fps"), path + ".inference.sample_fps")
        if not math.isclose(inference["sample_fps"], fps, rel_tol=1e-9, abs_tol=0):
            raise ValueError(f"{path}.inference.sample_fps: must match dataset.fps={fps}")
        registry[vehicle_id] = {
            "certificate_path": certificate_path,
            "checkpoint": identity,
            "inference": inference,
            "profiles": _certificate_profiles(certificate, path, inference),
        }
    return registry

def load_config(yaml_path: str) -> dict:
    """
    Loads a YAML file and returns its contents as a Python dictionary.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_vehicle_config(vehicle_key, vehicle_dict) -> dict:
    """
    Validates and returns a cleaned-up vehicle dictionary with all required fields.
    """
    
    if "type" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing required field 'type'."
        print(msg)
        raise ValueError(msg)

    vehicle_type = vehicle_dict["type"]
    if vehicle_type not in SUPPORTED_VEHICLE_TYPES:
        msg = (
            f"Vehicle '{vehicle_key}' has invalid type '{vehicle_type}'. "
            f"Must be one of {SUPPORTED_VEHICLE_TYPES}."
        )
        print(msg)
        raise ValueError(msg)

    # Required modules based on vehicle type
    required_modules = ["detector", "tracker", "predictor"]
    must_have_broadcaster = (vehicle_type == "broadcasting" or vehicle_type == "hybrid")
    if must_have_broadcaster:
        required_modules.append("broadcaster")
    must_have_listener = (vehicle_type == "aggregating" or vehicle_type == "hybrid")
    if must_have_listener:
        required_modules.append("listener")
    must_have_fusion = vehicle_type == "aggregating"


    for module in required_modules:
        if module not in vehicle_dict:
            msg = f"Vehicle '{vehicle_key}' of type '{vehicle_type}' must define '{module}'"
            print(msg)
            raise ValueError(msg)
            
            
#_________________________________________________________________________________________________

    # Validate 'parameters' block at vehicle level
    if "parameters" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing 'parameters' block."
        print(msg)
        raise ValueError(msg)

    params = vehicle_dict["parameters"]
    needed_params = [
        "tracking_buffer_s",
        "observed_past_s",
        "prediction_horizon_s",
        "device",
    ]
    for p in needed_params:
        if p not in params:
            msg = f"Vehicle '{vehicle_key}' parameters is missing '{p}'."
            print(msg)
            raise ValueError(msg)

    positive_params = [
        "tracking_buffer_s",
        "observed_past_s",
        "prediction_horizon_s",
    ]
    for p in positive_params:
        if not isinstance(params[p], (int, float)) or isinstance(params[p], bool) or not math.isfinite(params[p]) or params[p] <= 0:
            msg = f"Vehicle '{vehicle_key}' parameters.{p} must be a positive number."
            print(msg)
            raise ValueError(msg)

    if not isinstance(params["device"], str) or not params["device"].strip():
        msg = f"Vehicle '{vehicle_key}' parameters.device must be a non-empty string."
        print(msg)
        raise ValueError(msg)

#_________________________________________________________________________________________________

    # Validate 'detector'
    detector = vehicle_dict["detector"]
    if isinstance(detector, dict):
        if "name" not in detector:
            msg = f"Vehicle '{vehicle_key}' detector is missing 'name'."
            print(msg)
            raise ValueError(msg)
        det_name = detector["name"]
        if det_name not in SUPPORTED_DETECTORS:
            msg = (
                f"Vehicle '{vehicle_key}' detector.name='{det_name}' not in {SUPPORTED_DETECTORS}."
            )
            print(msg)
            raise ValueError(msg)
            
        # for detector you either need to provide detection and load them in wrapper
        # or you provide a checkpoint and in wrapper load from it
        # only if detector is gt, the checkpoint field can be omitted 
        if not(det_name == "gt" or det_name == "gt_occ"):
            if "detections" not in detector:
                msg = f"Vehicle '{vehicle_key}' detector, you must provide checkpoint or detections folder path)."
                print(msg)
                raise ValueError(msg)   
    else:
        msg = f"Vehicle '{vehicle_key}' detector must be a dictionary."
        print(msg)
        raise ValueError(msg)


#_________________________________________________________________________________________________


    # Validate 'tracker'
    tracker = vehicle_dict["tracker"]
    if isinstance(tracker, dict):
        if "name" not in tracker:
            msg = f"Vehicle '{vehicle_key}' tracker is missing 'name' field."
            print(msg)
            raise ValueError(msg)
        tracker_name = tracker["name"]
        if tracker_name not in SUPPORTED_TRACKERS:
            msg = (
                f"Vehicle '{vehicle_key}' tracker.name='{tracker_name}' invalid. "
                f"Must be one of {SUPPORTED_TRACKERS}."
            )
            print(msg)
            raise ValueError(msg)
    else:
        msg = f"Vehicle '{vehicle_key}' tracker must be a dictionary."
        print(msg)
        raise ValueError(msg)
        
#_________________________________________________________________________________________________


    # Validate 'predictor'
    predictor = vehicle_dict["predictor"]
    if not isinstance(predictor, dict):
        msg = f"Vehicle '{vehicle_key}' predictor must be a dictionary."
        print(msg)
        raise ValueError(msg)

    layout = predictor.get("layout")
    if layout not in SUPPORTED_PREDICTOR_LAYOUTS:
        msg = (
            f"Vehicle '{vehicle_key}' predictor.layout='{layout}' invalid. "
            f"Must be one of {SUPPORTED_PREDICTOR_LAYOUTS}."
        )
        print(msg)
        raise ValueError(msg)

    checkpoint = predictor.get("checkpoint")
    if checkpoint:
        if not isinstance(checkpoint, str) or not checkpoint.strip():
            msg = (
                f"Vehicle '{vehicle_key}' predictor.checkpoint must be "
                "a non-empty path."
            )
            print(msg)
            raise ValueError(msg)
    else:
        model_config = predictor.get("config")
        if not isinstance(model_config, str) or not model_config.strip():
            msg = (
                f"Vehicle '{vehicle_key}' predictor.config must be a non-empty "
                "TrajZoo model name when checkpoint is not provided."
            )
            print(msg)
            raise ValueError(msg)

        params_file_path = predictor.get("params_file_path")
        if params_file_path is not None and (
            not isinstance(params_file_path, str)
            or not params_file_path.strip()
        ):
            msg = (
                f"Vehicle '{vehicle_key}' predictor.params_file_path must be "
                "a non-empty path."
            )
            print(msg)
            raise ValueError(msg)

#_________________________________________________________________________________________________

    # Validate 'broadcaster' if must_have_broadcaster
    if must_have_broadcaster:
        broadcaster = vehicle_dict["broadcaster"]
        if not isinstance(broadcaster, dict):
            msg = f"Vehicle '{vehicle_key}' broadcaster must be a dictionary."
            print(msg)
            raise ValueError(msg)
        if "broadcasting_interval_s" not in broadcaster:
            msg = f"Vehicle '{vehicle_key}' broadcaster missing 'broadcasting_interval_s'."
            print(msg)
            raise ValueError(msg)
        broadcasting_interval_s = broadcaster["broadcasting_interval_s"]
        if not isinstance(broadcasting_interval_s, (int, float)) or isinstance(broadcasting_interval_s, bool) or broadcasting_interval_s <= 0:
            msg = f"Vehicle '{vehicle_key}' broadcaster.broadcasting_interval_s must be positive."
            print(msg)
            raise ValueError(msg)
        if "topic" not in broadcaster:
            msg = f"Vehicle '{vehicle_key}' broadcaster missing 'broadcasting topic'."
            print(msg)
            raise ValueError(msg)

#_________________________________________________________________________________________________

    # Validate 'listener' if must_have_listener
    if must_have_listener:
        listener = vehicle_dict["listener"]
        if not isinstance(listener, dict):
            msg = f"Vehicle '{vehicle_key}' listener must be a dictionary."
            print(msg)
            raise ValueError(msg)
        if "topic" not in listener:
            msg = f"Vehicle '{vehicle_key}' listener missing 'listener topic'."
            print(msg)
            raise ValueError(msg)  
        if "drop" in listener:
            if not isinstance(listener["drop"], bool):
                msg = f"Vehicle '{vehicle_key}' listener.drop must be a boolean."
                print(msg)
                raise ValueError(msg)
        else:
            vehicle_dict["listener"]["drop"] = False
            
        if "delay" in listener:
            delay = listener["delay"]
            if not isinstance(delay, dict):
                msg = f"Vehicle '{vehicle_key}' listener.delay must be a dictionary."
                print(msg)
                raise ValueError(msg)
    
            # allow partial specification; Listener will treat missing as zero-delay
            for key in ("k", "mu", "var"):
                if key in delay and not isinstance(delay[key], (int, float)):
                    msg = f"Vehicle '{vehicle_key}' listener.delay.{key} must be a number."
                    print(msg)
                    raise ValueError(msg)

#_________________________________________________________________________________________________

    # Each aggregation category owns its alignment, gate and fusion settings.
    if must_have_fusion:
        _validate_categories(vehicle_key, vehicle_dict)

#_________________________________________________________________________________________________


    # Validate 'sensors'
    if "sensors" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing 'sensors'."
        print(msg)
        raise ValueError(msg)
    if not isinstance(vehicle_dict["sensors"], list):
        msg = f"Vehicle '{vehicle_key}' sensors must be a list."
        print(msg)
        raise ValueError(msg)

    return vehicle_dict

def validate_dataset_block(dataset_block: dict) -> dict:
    if "path" not in dataset_block or "split" not in dataset_block or "fps" not in dataset_block:
        msg = "dataset block must contain 'path', 'split', and 'fps'."
        print(msg)
        raise ValueError(msg)
        
    dataset_path = dataset_block["path"]
    split = dataset_block["split"]
    fps = dataset_block["fps"]

    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        msg = "dataset.fps must be a positive number."
        print(msg)
        raise ValueError(msg)
    
    if split not in SPLITS:
        msg = (f"Dataset '{split}' is invalid, must be one of {SPLITS}.")
        print(msg)
        raise ValueError(msg)
    data = {}
    prefix_path = os.path.join(dataset_path, split)
    meta_file = os.path.join(prefix_path, "meta.txt")
    data_file = os.path.join(prefix_path, f"{split}_data.pkl")

    if os.path.isfile(data_file) and os.path.isfile(meta_file):
            data["data_file"] = data_file
            data["meta_file"] = meta_file
            data["name"] = dataset_block["name"]
            data["fps"] = float(fps)
    else:
        msg = (
            f"Either the file {data_file} or {meta_file} are missing, "
            f"make sure you run preprocess for {dataset_block['name']}."
        )
        print(msg)
        raise ValueError(msg)
    return data
    

def parse_config(config: dict) -> dict:
    """
    Parses and validates a DeepAccident YAML configuration, returning
    a dictionary with all parameters (dataset, ego_vehicle, vehicles, etc.).
    
    Raises ValueError if any required field is missing or invalid.
    """

    ########################## Validate dataset block
    if "dataset" not in config:
        msg = "Config must have a 'dataset' block."
        print(msg)
        raise ValueError(msg)
    dataset_block = config["dataset"]
    data = validate_dataset_block(dataset_block)

    ########################## Validate vehicles
    vehicles_dict = {}
    if "vehicles" in config:
        for vehicle_key, vehicle_val in config["vehicles"].items():
            validated_vehicle = validate_vehicle_config(vehicle_key, vehicle_val)
            vehicles_dict[vehicle_key] = validated_vehicle
    else:
        msg = "No 'vehicles' block found"
        print(msg)
        raise ValueError(msg)

    
    ########################## Validate ego_vehicle
    if "ego_vehicle" not in config:
        msg = "Config must have an 'ego_vehicle' block."
        print(msg)
        raise ValueError(msg)
    vehicles = set(vehicles_dict.keys())
    if config["ego_vehicle"] not in vehicles:
        msg = "Ego-vehicle must be from the list of the vehicles"
        print(msg)
        raise ValueError(msg)

    ########################## Parsed config 
    parsed_config = {
        "data": data,
        "ego_vehicle": config["ego_vehicle"],
        "vehicles": vehicles_dict,
        "calibration_by_vehicle": _load_calibration_by_vehicle(
            vehicles_dict, data["fps"], config["ego_vehicle"],
        ),
    }
    print("DeepAccident configuration parsed successfully.")
    return parsed_config
