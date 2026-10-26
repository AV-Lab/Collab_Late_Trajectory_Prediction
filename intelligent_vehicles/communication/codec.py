"""Shared covariance and squared-error risk representation for packets."""

import numpy as np


PACKET_SCHEMA_VERSION = 2
RISK_UNIT = "m2"
POSITION_SCALE = 100.0
COVARIANCE_SCALE = 4096.0
CORRELATION_LIMIT = 0.999
POSITION_QUANTIZATION_VARIANCE = 2.0 * ((1.0 / POSITION_SCALE) ** 2 / 12.0)


def pack_covariances(covariance_list):
    """Encode [variance_x, covariance_xy, variance_y] as int16 triples."""
    covariance = np.asarray(covariance_list, dtype=np.float64)
    if covariance.ndim != 2 or covariance.shape[1] != 3:
        raise ValueError("Covariance entries must have shape [T, 3].")
    if not np.isfinite(covariance).all():
        raise ValueError("Covariance entries must contain only finite values.")

    variance_x = covariance[:, 0]
    covariance_xy = covariance[:, 1]
    variance_y = covariance[:, 2]
    determinant = variance_x * variance_y - covariance_xy ** 2
    if np.any(variance_x <= 0.0) or np.any(variance_y <= 0.0) or np.any(determinant <= 0.0):
        raise ValueError("Covariance entries must be positive definite.")

    std_x = np.sqrt(variance_x)
    std_y = np.sqrt(variance_y)
    correlation = np.clip(covariance_xy / (std_x * std_y),
                          -CORRELATION_LIMIT, CORRELATION_LIMIT)
    transformed = np.column_stack((np.log(std_x), np.log(std_y),
                                   np.arctanh(correlation)))
    quantized = np.rint(transformed * COVARIANCE_SCALE)
    limits = np.iinfo(np.int16)
    if np.any(quantized < limits.min) or np.any(quantized > limits.max):
        raise ValueError("Covariance exceeds the supported quantization range.")
    return quantized.astype("<i2").tobytes()


def decode_covariances(data, count):
    """Reconstruct receiver covariance, adding position variance exactly once."""
    if (isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer)) or count <= 0):
        raise ValueError("Prediction count must be a positive integer.")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError("Packed covariances must be bytes.")
    values = np.frombuffer(data, dtype="<i2")
    if values.size != 3 * count:
        raise ValueError("Packed covariance count does not match prediction count.")
    transformed = values.reshape(count, 3).astype(np.float64) / COVARIANCE_SCALE
    std_x = np.exp(transformed[:, 0])
    std_y = np.exp(transformed[:, 1])
    correlation = np.tanh(transformed[:, 2])
    covariance = np.empty((count, 2, 2), dtype=np.float64)
    covariance[:, 0, 0] = std_x ** 2 + POSITION_QUANTIZATION_VARIANCE
    covariance[:, 1, 1] = std_y ** 2 + POSITION_QUANTIZATION_VARIANCE
    covariance[:, 0, 1] = correlation * std_x * std_y
    covariance[:, 1, 0] = covariance[:, 0, 1]
    return covariance


def validate_bound_pair(pair):
    """Decode [lower, upper] in m²; unavailable native points remain None."""
    if pair is None:
        return None
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        raise ValueError("Risk bounds must be [lower, upper] pairs or null.")
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, float, np.integer, np.floating))
           or not np.isfinite(value) for value in pair):
        raise ValueError("Risk bounds must be finite numeric values.")
    lower, upper = map(float, pair)
    if not 0.0 <= lower <= upper:
        raise ValueError("Risk bounds require 0 <= lower <= upper.")
    return {"lower": lower, "upper": upper}


def quantized_risk_bounds(bounds, displacement):
    """Enlarge native RMS bounds by the encoded mean's displacement."""
    if bounds is None:
        return None
    pair = validate_bound_pair([bounds["lower"], bounds["upper"]])
    if (isinstance(displacement, (bool, np.bool_))
            or not isinstance(displacement, (int, float, np.integer, np.floating))
            or not np.isfinite(displacement) or displacement < 0.0):
        raise ValueError("Mean quantization displacement must be finite and nonnegative.")
    return {
        "lower": max(0.0, np.sqrt(pair["lower"]) - displacement) ** 2,
        "upper": (np.sqrt(pair["upper"]) + displacement) ** 2,
    }
