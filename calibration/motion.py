"""Fixed TrajZoo GUI motion tags for original forecasts in Waymo coordinates.

Calibration and sending vehicles use the same observed-history/prediction input.
Ground-truth futures and aligned forecasts never enter group assignment.
"""

import math

import numpy as np


MOTION_TAGS = (
    "stationary", "straight", "turn_left_mild", "turn_left_sharp",
    "turn_right_mild", "turn_right_sharp", "u_turn", "merge_left", "merge_right",
)
MOTION_DEFINITION = {
    "version": 1,
    "tags": list(MOTION_TAGS),
    "coordinate_convention": "waymo_v1",
    "input_policy": "last_obs_len_history_including_current_plus_original_prediction",
    "heading_policy": "first_and_last_displacement_at_least_0.1m",
    "thresholds": {
        "stationary_min_displacement": 2.0,
        "straight_max_turn_deg": 20.0,
        "mild_max_turn_deg": 60.0,
        "uturn_min_turn_deg": 150.0,
        "merge_min_lateral_offset": 2.0,
        "min_points": 3,
        "heading_epsilon_meters": 0.1,
    },
}


def classify_motion(history, prediction, obs_len):
    """Return one native-forecast tag, or None for incomplete/invalid inputs."""
    if (isinstance(obs_len, (bool, np.bool_))
            or not isinstance(obs_len, (int, np.integer)) or obs_len < 2):
        return None
    try:
        history = np.asarray(history, dtype=np.float64)
        prediction = np.asarray(prediction, dtype=np.float64)
        if (history.ndim != 2 or history.shape[1] < 2 or len(history) < obs_len
                or prediction.ndim != 2 or prediction.shape[1] < 2 or not len(prediction)):
            return None
        points = np.concatenate((history[-obs_len:, :2], prediction[:, :2]))
        if not np.isfinite(points).all():
            return None
        with np.errstate(over="raise", invalid="raise"):
            initial = points[1:] - points[0]
            final = points[-1] - points[-2::-1]
            displacement = points[-1] - points[0]
            if math.hypot(*displacement) < 2.0:
                return "stationary"
            headings = []
            for differences in (initial, final):
                eligible = next((delta for delta in differences if math.hypot(*delta) >= 0.1), None)
                headings.append(math.atan2(eligible[1], eligible[0]) if eligible is not None else 0.0)
            turn = (math.degrees(headings[1] - headings[0]) + 180.0) % 360.0 - 180.0
            if turn <= -180.0:
                turn = 180.0
            lateral = -math.sin(headings[0]) * displacement[0] + math.cos(headings[0]) * displacement[1]
            magnitude = abs(turn)
            if magnitude >= 150.0:
                return "u_turn"
            if magnitude > 60.0:
                return "turn_left_sharp" if turn > 0.0 else "turn_right_sharp"
            if magnitude >= 20.0:
                return "turn_left_mild" if turn > 0.0 else "turn_right_mild"
            if abs(lateral) >= 2.0:
                return "merge_left" if lateral > 0.0 else "merge_right"
            return "straight"
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return None
