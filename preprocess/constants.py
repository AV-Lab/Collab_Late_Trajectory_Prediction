class Constants:
    # Shared preprocessing parameters
    OCCLUSION_RAYS_K = 31
    OCCLUSION_EPS = 1e-2
    OCCLUSION_VERTICAL_CHECK = True
    YAW_IN_DEGREES = False
    KEEP_RADIUS_M = 50.0

    # DeepAccident
    DEEPACCIDENT_CAMERA_SENSORS = [
        "Camera_FrontLeft",
        "Camera_Front",
        "Camera_FrontRight",
        "Camera_BackLeft",
        "Camera_Back",
        "Camera_BackRight",
    ]
    DEEPACCIDENT_LIDAR_SENSOR = "lidar01"
    DEEPACCIDENT_AGENTS = [
        "ego_vehicle",
        "ego_vehicle_behind",
        "other_vehicle",
        "other_vehicle_behind",
    ]
    DEEPACCIDENT_FPS = 10
    DEEPACCIDENT_STEP = 1.0 / DEEPACCIDENT_FPS

    # OPV2V
    OPV2V_FPS = 10
    OPV2V_STEP = 1.0 / OPV2V_FPS

    # V2V4Real
    V2V4REAL_FPS = 10
    V2V4REAL_STEP = 1.0 / V2V4REAL_FPS
