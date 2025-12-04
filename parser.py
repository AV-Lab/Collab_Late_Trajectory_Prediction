import os
import yaml
import logging

SUPPORTED_VEHICLE_TYPES = {"basic", "aggregating", "broadcasting", "hybrid"}
SUPPORTED_DETECTORS = {"gt", "gt_occ", "centerpoint"}
SUPPORTED_PREDICTORS = {"lstm_nll", "transformer_nll"}
SUPPORTED_TRACKERS = {"gt", "ab3dmot"}
SPLITS = {"train", "valid", "test"}

def load_config(yaml_path: str) -> dict:
    """
    Loads a YAML file and returns its contents as a Python dictionary.
    """
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_vehicle_config(vehicle_key, vehicle_dict, logger: logging.Logger) -> dict:
    """
    Validates and returns a cleaned-up vehicle dictionary with all required fields.
    """
    
    if "type" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing required field 'type'."
        logger.error(msg)
        raise ValueError(msg)

    vehicle_type = vehicle_dict["type"]
    if vehicle_type not in SUPPORTED_VEHICLE_TYPES:
        msg = (
            f"Vehicle '{vehicle_key}' has invalid type '{vehicle_type}'. "
            f"Must be one of {SUPPORTED_VEHICLE_TYPES}."
        )
        logger.error(msg)
        raise ValueError(msg)

    # Required modules based on vehicle type
    required_modules = ["detector", "tracker", "predictor"]
    must_have_broadcaster = (vehicle_type == "broadcasting" or vehicle_type == "hybrid")
    if must_have_broadcaster:
        required_modules.append("broadcaster")
    must_have_listener = (vehicle_type == "aggregating" or vehicle_type == "hybrid")
    if must_have_listener:
        required_modules.append("listener")


    for module in required_modules:
        if module not in vehicle_dict:
            msg = f"Vehicle '{vehicle_key}' of type '{vehicle_type}' must define '{module}'"
            logger.error(msg)
            raise ValueError(msg)
            
            
#_________________________________________________________________________________________________

    # Validate 'parameters' block at vehicle level
    if "parameters" not in vehicle_dict:
        msg = f"Vehicle '{vehicle_key}' is missing 'parameters' block."
        logger.error(msg)
        raise ValueError(msg)

    params = vehicle_dict["parameters"]
    needed_params = ["prediction_frequency", "device", "prediction_horizon"]
    for p in needed_params:
        if p not in params:
            msg = f"Vehicle '{vehicle_key}' parameters is missing '{p}'."
            logger.error(msg)
            raise ValueError(msg)

#_________________________________________________________________________________________________

    # Validate 'detector'
    detector = vehicle_dict["detector"]
    if isinstance(detector, dict):
        if "name" not in detector:
            msg = f"Vehicle '{vehicle_key}' detector is missing 'name'."
            logger.error(msg)
            raise ValueError(msg)
        det_name = detector["name"]
        if det_name not in SUPPORTED_DETECTORS:
            msg = (
                f"Vehicle '{vehicle_key}' detector.name='{det_name}' not in {SUPPORTED_DETECTORS}."
            )
            logger.error(msg)
            raise ValueError(msg)
            
        # for detector you either need to provide detection and load them in wrapper
        # or you provide a checkpoint and in wrapper load from it
        # only if detector is gt, the checkpoint field can be omitted 
        if not(det_name == "gt" or det_name == "gt_occ"):
            if "det_path" not in detector:
                msg = f"Vehicle '{vehicle_key}' detector, you must provide checkpoint or detections folder path)."
                logger.error(msg)
                raise ValueError(msg)   
    else:
        msg = f"Vehicle '{vehicle_key}' detector must be a dictionary."
        logger.error(msg)
        raise ValueError(msg)


#_________________________________________________________________________________________________


    # Validate 'tracker'
    tracker = vehicle_dict["tracker"]
    if isinstance(tracker, dict):
        if "name" not in tracker:
            msg = f"Vehicle '{vehicle_key}' tracker is missing 'name' field."
            logger.error(msg)
            raise ValueError(msg)
        tracker_name = tracker["name"]
        if tracker_name not in SUPPORTED_TRACKERS:
            msg = (
                f"Vehicle '{vehicle_key}' tracker.name='{tracker_name}' invalid. "
                f"Must be one of {SUPPORTED_TRACKERS}."
            )
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"Vehicle '{vehicle_key}' tracker must be a dictionary."
        logger.error(msg)
        raise ValueError(msg)
        
#_________________________________________________________________________________________________


    # Validate 'predictor'
    predictor = vehicle_dict["predictor"]
    if isinstance(predictor, dict):
        if "name" not in predictor:
            msg = f"Vehicle '{vehicle_key}' predictor is missing 'name'."
            logger.error(msg)
            raise ValueError(msg)
        pred_name = predictor["name"]
        if pred_name not in SUPPORTED_PREDICTORS:
            msg = (
                f"Vehicle '{vehicle_key}' predictor.name='{pred_name}' invalid. "
                f"Must be one of {SUPPORTED_PREDICTORS}."
            )
            logger.error(msg)
            raise ValueError(msg)
        if "checkpoint" not in predictor:
            msg = f"Vehicle '{vehicle_key}' predictor is in eval mode, you must provide checkpoint)."
            logger.error(msg)
            raise ValueError(msg)
    else:
        msg = f"Vehicle '{vehicle_key}' predictor must be a dictionary."
        logger.error(msg)
        raise ValueError(msg)

#_________________________________________________________________________________________________

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
        if "topic" not in broadcaster:
            msg = f"Vehicle '{vehicle_key}' broadcaster missing 'broadcasting topic'."
            logger.error(msg)
            raise ValueError(msg)

#_________________________________________________________________________________________________

    # Validate 'listener' if must_have_listener
    if must_have_listener:
        listener = vehicle_dict["listener"]
        if not isinstance(listener, dict):
            msg = f"Vehicle '{vehicle_key}' listener must be a dictionary."
            logger.error(msg)
            raise ValueError(msg)
        if "topic" not in listener:
            msg = f"Vehicle '{vehicle_key}' listener missing 'listener topic'."
            logger.error(msg)
            raise ValueError(msg)            
#_________________________________________________________________________________________________


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

def validate_dataset_block(dataset_block: dict, logger: logging.Logger) -> dict:
    if "path" not in dataset_block or "split" not in dataset_block:
        msg = "dataset block must contain 'path' and 'prefixes'."
        logger.error(msg)
        raise ValueError(msg)
        
    dataset_path = dataset_block["path"]
    split = dataset_block["split"]
    
    if split not in SPLITS:
        msg = (f"Dataset '{split}' is invalid, must be one of {SPLITS}.")
        logger.error(msg)
        raise ValueError(msg)
    data = {}
    prefix_path = os.path.join(dataset_path, split)
    meta_file = os.path.join(prefix_path, "meta.txt")
    data_file = os.path.join(prefix_path, f"{split}_data.pkl")

    if os.path.isfile(data_file) and os.path.isfile(meta_file):
            data["data_file"] = data_file
            data["meta_file"] = meta_file
            data["name"] = dataset_block["name"]
    else:
        msg = (
            f"Either the file {data_file} or {meta_file} are missing, "
            f"make sure you run preprocess for {dataset_block['name']}."
        )
        logger.error(msg)
        raise ValueError(msg)
    logger.debug(f"Dataset block validated: {data}")
    
    if "preprocessed" in dataset_block and dataset_block["preprocessed"]:
        data["preprocessed"] = True
        msg = ("You specified preprocessed tag, there WILL BE NO DATA PROCESSING "
               "into individual vehcile observation during vehicles initialization."
        )
        logger.info(msg)
    else:
        data["preprocessed"] = False
        
    return data
    

def parse_config(config: dict, logger: logging.Logger) -> dict:
    """
    Parses and validates a DeepAccident YAML configuration, returning
    a dictionary with all parameters (dataset, ego_vehicle, vehicles, etc.).
    
    Raises ValueError if any required field is missing or invalid.
    """

    ########################## Validate dataset block
    if "dataset" not in config:
        msg = "Config must have a 'dataset' block."
        logger.error(msg)
        raise ValueError(msg)
    dataset_block = config["dataset"]
    data = validate_dataset_block(dataset_block, logger)

    ########################## Validate vehicles
    vehicles_dict = {}
    if "vehicles" in config:
        for vehicle_key, vehicle_val in config["vehicles"].items():
            validated_vehicle = validate_vehicle_config(vehicle_key, vehicle_val, logger)
            vehicles_dict[vehicle_key] = validated_vehicle
        logger.debug(f"vehicles validated: {vehicles_dict}")
    else:
        msg = "No 'vehicles' block found"
        logger.error(msg)
        raise ValueError(msg)

    
    ########################## Validate ego_vehicle
    if "ego_vehicle" not in config:
        msg = "Config must have an 'ego_vehicle' block."
        logger.error(msg)
        raise ValueError(msg)
    vehicles = set(vehicles_dict.keys())
    if config["ego_vehicle"] not in vehicles:
        msg = "Ego-vehicle must be from the list of the vehicles"
        logger.error(msg)
        raise ValueError(msg)
    else:    
        logger.debug(f"ego_vehicle validated: {config['ego_vehicle']} is ego")

    ########################## Parsed config 
    parsed_config = {
        "data": data,
        "ego_vehicle": config["ego_vehicle"],
        "vehicles": vehicles_dict
    }
    logger.info("DeepAccident configuration parsed successfully.")
    return parsed_config