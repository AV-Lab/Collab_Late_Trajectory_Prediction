"""World-frame geometry shared by the visualization renderers."""

import numpy as np


def box_corners(box):
    """Return four rotated BEV corners for [x, y, z, length, width, height, yaw]."""
    box = np.asarray(box, dtype=float).reshape(-1)
    if box.size < 7 or not np.isfinite(box[:7]).all() or np.any(box[3:6] <= 0):
        return np.empty((0, 2))
    length, width, yaw = box[3], box[4], box[6]
    corners = np.array([[length, width], [length, -width],
                        [-length, -width], [-length, width]]) / 2
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    return corners @ rotation.T + box[:2]


def covariance_ellipse(center, cov, sigma_scale=1.0, n_points=48):
    """Return a closed standard-deviation ellipse; invalid matrices return None."""
    if cov is None:
        return None
    cov = np.asarray(cov, dtype=float)
    center = np.asarray(center, dtype=float)
    if cov.shape != (2, 2) or center.shape != (2,) or not np.isfinite(cov).all():
        return None
    if not np.isfinite(center).all() or not np.allclose(cov, cov.T, atol=1e-8):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh((cov + cov.T) / 2)
    if eigenvalues[0] < -1e-10:
        return None
    angles = np.linspace(0, 2 * np.pi, n_points + 1)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    radii = sigma_scale * np.sqrt(np.maximum(eigenvalues, 0))
    return (circle * radii) @ eigenvectors.T + center


def world_to_bev(xy, origin):
    """Translate world XY into an ego-centred view without changing orientation."""
    return np.asarray(xy, dtype=float)[..., :2] - np.asarray(origin, dtype=float)[:2]


def lidar_world_xy(points, calibration):
    """Apply the full native LiDAR-to-world transform, then project into XY."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("LiDAR points must have shape [N, 3] or [N, 4].")
    transform = (np.asarray(calibration["ego_to_world"], dtype=float)
                 @ np.asarray(calibration["lidar_to_ego"], dtype=float))
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("LiDAR visualization requires a finite 4x4 world transform.")
    xyz = points[:, :3]
    return (xyz @ transform[:3, :3].T + transform[:3, 3])[:, :2]
