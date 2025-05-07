import os
import yaml
import logging
import sys


VALID_VEHICLE_TYPES = {"basic", "aggregating", "broadcasting", "hybrid"}
VALID_DETECTORS = {"gt", "centerpoint", "pointpillars"}
VALID_PREDICTORS = {"lstm", "gcn", "gat", "gcn_temporal", "gat_temporal", "transformer", "transformer_gnn"}
VALID_MODES = {"train", "eval"}
VALID_FPS = [10, 20]


def load_config(yaml_path: str) -> dict:
    """
    Loads a YAML file and returns its contents as a Python dictionary.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_vehicle_config(vehicle_dict: dict, vehicle_key: str, logger: logging.Logger) -> dict:
    """
    Validates and returns a cleaned-up vehicle dictionary with all required fields.
    Raises ValueError if any required field is missing or invalid.
    
    vehicle_dict: e.g. from the YAML "ego_vehicle" or "vehicles.vehicle_1".
    vehicle_key: string name used for error messaging.
    logger: logger instance for logging validation messages.
    """

    # name & type
    if "name" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing required field 'name'."
        logger.error(msg)
        raise ValueError(msg)

    if "type" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing required field 'type'."
        logger.error(msg)
        raise ValueError(msg)

    vehicle_type = vehicle_dict["type"]
    if vehicle_type not in VALID_VEHICLE_TYPES:
        msg = (
            f"Vehicle '{vehicle_key}' has invalid type '{vehicle_type}'. "
            f"Must be one of {VALID_VEHICLE_TYPES}."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Required modules based on vehicle type
    must_have_broadcaster = (vehicle_type == "broadcasting" or vehicle_type == "hybrid")
    required_modules = ["detector", "tracker", "predictor"]
    if must_have_broadcaster:
        required_modules.append("broadcaster")

    for module in required_modules:
        if module not in vehicle_dict:
            msg = f"Vehicle '{vehicle_key}' of type '{vehicle_type}' must define '{module}'"
            logger.error(msg)
            raise ValueError(msg)

    # Validate 'parameters' block at vehicle level
    if "parameters" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing 'parameters' block."
        logger.error(msg)
        raise ValueError(msg)

    params = vehicle_dict["parameters"]
    needed_params = ["fps", "prediction_horizon", "prediction_frequency", "forecasting_frequency"]
    for p in needed_params:
        if p not in params:
            msg = f"Vehicle '{vehicle_key}' parameters is missing '{p}'."
            logger.error(msg)
            raise ValueError(msg)

    # Optionally check valid FPS
    if params["fps"] not in VALID_FPS:
        msg = (
            f"Vehicle '{vehicle_key}' has fps '{params['fps']}' not in {VALID_FPS}."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Validate 'detector'
    detector = vehicle_dict["detector"]
    if isinstance(detector, dict):
        if "name" not in detector:
            msg = f"Vehicle '{vehicle_key}' detector is missing 'name'."
            logger.error(msg)
            raise ValueError(msg)
        det_name = detector["name"]
        if det_name not in VALID_DETECTORS:
            msg = (
                f"Vehicle '{vehicle_key}' detector.name='{det_name}' not in {VALID_DETECTORS}."
            )
            logger.error(msg)
            raise ValueError(msg)

        if "mode" in detector:
            if detector["mode"] not in VALID_MODES:
                msg = (
                    f"Vehicle '{vehicle_key}' detector.mode='{detector['mode']}' invalid. "
                    f"Must be one of {VALID_MODES}."
                )
                logger.error(msg)
                raise ValueError(msg)

    else:
        # If it's just a string
        det_name = detector
        if det_name not in VALID_DETECTORS:
            msg = (
                f"Vehicle '{vehicle_key}' detector='{det_name}' not in {VALID_DETECTORS}."
            )
            logger.error(msg)
            raise ValueError(msg)
        # Convert to dict for consistency
        vehicle_dict["detector"] = {"name": det_name}

    # Validate 'tracker'
    tracker = vehicle_dict["tracker"]
    if isinstance(tracker, dict):
        if "class" not in tracker:
            msg = f"Vehicle '{vehicle_key}' tracker is missing 'class' field."
            logger.error(msg)
            raise ValueError(msg)
        if tracker["class"] != "AB3DMOT":
            msg = (
                f"Vehicle '{vehicle_key}' tracker class must be 'AB3DMOT', got '{tracker['class']}'"
            )
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"Vehicle '{vehicle_key}' tracker must be a dictionary."
        logger.error(msg)
        raise ValueError(msg)

    # Validate 'predictor'
    predictor = vehicle_dict["predictor"]
    if isinstance(predictor, dict):
        if "name" not in predictor:
            msg = f"Vehicle '{vehicle_key}' predictor is missing 'name'."
            logger.error(msg)
            raise ValueError(msg)
        pred_name = predictor["name"]
        if pred_name not in VALID_PREDICTORS:
            msg = (
                f"Vehicle '{vehicle_key}' predictor.name='{pred_name}' invalid. "
                f"Must be one of {VALID_PREDICTORS}."
            )
            logger.error(msg)
            raise ValueError(msg)
        # predictor mode
        if "mode" not in predictor:
            msg = f"Vehicle '{vehicle_key}' predictor is missing 'mode' (train/eval)."
            logger.error(msg)
            raise ValueError(msg)
        if predictor["mode"] not in VALID_MODES:
            msg = (
                f"Vehicle '{vehicle_key}' predictor.mode='{predictor['mode']}' invalid. "
                f"Must be one of {VALID_MODES}."
            )
            logger.error(msg)
            raise ValueError(msg)
        # 'checkpoint' is optional, but you can enforce or not
        # e.g. if predictor["mode"] == "train" => must have checkpoint, etc.
    else:
        msg = f"Vehicle '{vehicle_key}' predictor must be a dictionary."
        logger.error(msg)
        raise ValueError(msg)

    # Validate 'broadcaster' if must_have_broadcaster
    if must_have_broadcaster:
        broadcaster = vehicle_dict["broadcaster"]
        if not isinstance(broadcaster, dict):
            msg = f"Vehicle '{vehicle_key}' broadcaster must be a dictionary."
            logger.error(msg)
            raise ValueError(msg)
        if "broadcasting_frequency" not in broadcaster:
            msg = f"Vehicle '{vehicle_key}' broadcaster missing 'broadcasting_frequency'."
            logger.error(msg)
            raise ValueError(msg)
        if "broadcasting_range" not in broadcaster:
            msg = f"Vehicle '{vehicle_key}' broadcaster missing 'broadcasting_range'."
            logger.error(msg)
            raise ValueError(msg)

    # Validate 'sensors'
    if "sensors" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing 'sensors'."
        logger.error(msg)
        raise ValueError(msg)
    if not isinstance(vehicle_dict["sensors"], list):
        msg = f"Vehicle '{vehicle_key}' sensors must be a list."
        logger.error(msg)
        raise ValueError(msg)

    logger.debug(f"Vehicle '{vehicle_key}' validated successfully.")
    return vehicle_dict


###############################################################################
# PARSE DEEPACCIDENT CONFIG
###############################################################################
def parse_deepaccident_config(config: dict, logger: logging.Logger) -> dict:
    """
    Parses and validates a DeepAccident YAML configuration, returning
    a dictionary with all parameters (dataset, ego_vehicle, vehicles, etc.).
    
    Raises ValueError if any required field is missing or invalid.
    """

    # 1) Validate dataset block
    if "dataset" not in config:
        msg = "Config must have a 'dataset' block."
        logger.error(msg)
        raise ValueError(msg)
    dataset_block = config["dataset"]
    if "path" not in dataset_block or "prefixes" not in dataset_block:
        msg = "dataset block must contain 'path' and 'prefixes'."
        logger.error(msg)
        raise ValueError(msg)

    dataset_path = dataset_block["path"]
    prefixes = dataset_block["prefixes"]
    data = {}

    for prefix in prefixes:
        prefix_path = os.path.join(dataset_path, prefix)
        pickle_file = os.path.join(prefix_path, f"{prefix}_data.pkl")
        if os.path.isfile(pickle_file):
            data[prefix] = pickle_file
        else:
            msg = (
                f"Either you specified prefix '{prefix}' which is out of dataset folder, "
                f"or you did not run preprocess for '{prefix}' in {dataset_path}."
            )
            logger.error(msg)
            raise ValueError(msg)
    logger.debug(f"Dataset block validated: {data}")

    # 2) Validate ego_vehicle
    if "ego_vehicle" not in config:
        msg = "Config must have an 'ego_vehicle' block."
        logger.error(msg)
        raise ValueError(msg)
    ego_vehicle_block = validate_vehicle_config(config["ego_vehicle"], "ego_vehicle", logger)
    logger.debug(f"ego_vehicle validated: {ego_vehicle_block}")

    # 3) Validate vehicles
    vehicles_dict = {}
    if "vehicles" in config:
        for vehicle_key, vehicle_val in config["vehicles"].items():
            validated_vehicle = validate_vehicle_config(vehicle_val, vehicle_key, logger)
            vehicles_dict[vehicle_key] = validated_vehicle
        logger.debug(f"vehicles validated: {vehicles_dict}")
    else:
        logger.info("No 'vehicles' block found. Proceeding with empty vehicles list.")

    # 4) Assemble final
    parsed_config = {
        "data": data,
        "ego_vehicle": ego_vehicle_block,
        "vehicles": vehicles_dict
    }
    logger.info("DeepAccident configuration parsed successfully.")
    return parsed_config


###############################################################################
# PARSE OPV2V CONFIG
###############################################################################
def parse_opv2v_config(yaml_path: str, logger: logging.Logger) -> dict:
    # Future implementation
    pass


###############################################################################
# PARSE V2V4REAL CONFIG
###############################################################################
def parse_v2v4real_config(yaml_path: str, logger: logging.Logger) -> dict:
    # Future implementation
    pass