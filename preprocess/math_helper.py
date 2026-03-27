from __future__ import annotations
from typing import Iterable, Tuple
import math
import numpy as np

# ------------------------------- simple trig/SE(3) -------------------------------

def cosd(a_deg: float) -> float:
    return math.cos(math.radians(a_deg))

def sind(a_deg: float) -> float:
    return math.sin(math.radians(a_deg))

def rpy_to_R(roll: float, yaw: float, pitch: float, degrees: bool = True) -> np.ndarray:
    if degrees:
        roll, yaw, pitch = map(math.radians, (roll, yaw, pitch))
    cr, sr = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float64)
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float64)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float64)
    return (Rz @ Ry) @ Rx

def pose_to_T(x: float, y: float, z: float, roll: float, yaw: float, pitch: float, degrees: bool = True) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = rpy_to_R(roll, yaw, pitch, degrees=degrees)
    T[:3, 3] = np.array([x, y, z], dtype=np.float64)
    return T

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]
    t = T[:3,3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3,:3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti

# ------------------------------- planar BEV geometry ------------------------------

def rot2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def box_corners_xy(cx: float, cy: float, length: float, width: float, yaw_rad: float) -> np.ndarray:
    """Return 4x2 corners (x,y) of an oriented rectangle."""
    hl, hw = 0.5 * length, 0.5 * width
    local = np.array([[ hl,  hw], [ hl, -hw], [-hl, -hw], [-hl,  hw]], dtype=np.float64)
    R = rot2d(yaw_rad)
    return (local @ R.T) + np.array([cx, cy], dtype=np.float64)

def angles_from(origin: np.ndarray, pts: np.ndarray) -> np.ndarray:
    v = pts - origin[None, :]
    return np.arctan2(v[:, 1], v[:, 0])

def min_covering_arc(angles: np.ndarray) -> Tuple[float, float]:
    """Smallest unwrapped arc [L,R] covering all input angles (wrap-aware)."""
    a = np.sort((angles + np.pi) % (2 * np.pi)) - np.pi
    diffs = np.diff(np.r_[a, a[0] + 2 * np.pi])
    i = int(np.argmax(diffs))
    L = a[(i + 1) % len(a)]
    R = L + (2 * np.pi - diffs[i])
    return L, R

def rect_angular_span(origin: np.ndarray, corners: np.ndarray) -> Tuple[float, float]:
    return min_covering_arc(angles_from(origin, corners))

def ray_segment_intersection(s: np.ndarray, d: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> float | None:
    """Ray s + t d (t>=0) with segment p0->p1. Returns t or None."""
    v = p1 - p0
    M = np.array([[d[0], -v[0]], [d[1], -v[1]]], dtype=np.float64)
    b = p0 - s
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    if abs(det) < 1e-12:
        return None
    inv = np.array([[ M[1,1], -M[0,1]], [-M[1,0],  M[0,0]]], dtype=np.float64) / det
    t, u = (inv @ b)
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return float(t)
    return None

def ray_rect_distance(s: np.ndarray, theta: float, corners: np.ndarray) -> float | None:
    """First intersection distance from origin to rectangle edges; None if miss."""
    d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    ts = []
    for i in range(4):
        p0 = corners[i]
        p1 = corners[(i + 1) % 4]
        t = ray_segment_intersection(s, d, p0, p1)
        if t is not None:
            ts.append(t)
    return float(min(ts)) if ts else None

def bounds_from_corners(all_corners: Iterable[np.ndarray]) -> Tuple[float, float, float, float]:
    xs, ys = [], []
    for cs in all_corners:
        xs.extend(cs[:, 0].tolist())
        ys.extend(cs[:, 1].tolist())
    return (min(xs), max(xs), min(ys), max(ys)) if xs else (0.0, 0.0, 0.0, 0.0)
