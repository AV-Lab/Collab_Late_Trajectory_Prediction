#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
import open3d as o3d

from visualization.bbox_visualize import BBoxVisualizer
from visualization.utils import force_camera_pose

logger = logging.getLogger(__name__)


class TrackerVisualizer(BBoxVisualizer):
    """
    Trajectory + box visualiser.
    • If transform_to_global is True   →  LiDAR → world conversion applied
    • If transform_to_global is False →  data drawn in LiDAR frame
    """

    def __init__(self, camera_height=40.0, zoom_level=0.15):
        super().__init__(camera_height, zoom_level)
        self.trajectory_geometries = []

    # ------------------------------------------------ trajectory (unchanged)
    def update_trajectories(self, tracklets):
        for tr in tracklets:
            tracklet = tr["tracklet"]
            if len(tracklet) < 2:
                continue
            label = tr["category"]
            if label not in self.label_colors:
                self.label_colors[label] = np.random.rand(3)
            color = self.label_colors[label]

            pts = np.array([(p.x, p.y, p.z) for p in tracklet])
            lines = [[i, i + 1] for i in range(len(pts) - 1)]
            ls = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(pts),
                lines=o3d.utility.Vector2iVector(lines),
            )
            ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
            self.vis.add_geometry(ls); self.trajectory_geometries.append(ls)

            for p in pts:
                dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
                dot.translate(np.asarray(p) - np.array([0, 0, 0.1]))
                dot.paint_uniform_color(color)
                self.vis.add_geometry(dot); self.trajectory_geometries.append(dot)

    # ------------------------------------------------ helpers
    @staticmethod
    def _lidar_to_world(calib):
        return calib["ego_to_world"] @ calib["lidar_to_ego"]

    # ------------------------------------------------ cloud
    def update_cloud(self, pc_dict, calib, to_global: bool):
        xyz = pc_dict["data"][:, :3].astype(np.float32)

        if to_global:
            T = self._lidar_to_world(calib)
            xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
            xyz = (T @ xyz_h.T).T[:, :3]

        self.cloud.points = o3d.utility.Vector3dVector(xyz)
        self.cloud.colors = o3d.utility.Vector3dVector(
            np.tile([[1, 1, 1]], (xyz.shape[0], 1))
        )
        self.vis.update_geometry(self.cloud)

    # ------------------------------------------------ main
    def visualize_tracks(
        self,
        point_cloud,
        tracklets,
        ego_pose,
        calib,
        transform_to_global: bool = True,
    ):
        """Draw world-frame view if flag True, otherwise LiDAR frame."""
        self.update_cloud(point_cloud, calib, transform_to_global)

        # clear previous geometries
        for g in self.bbox_geometries + self.trajectory_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.bbox_geometries.clear(); self.trajectory_geometries.clear()

        # ------------------------------------------------ ego pose & boxes
        if transform_to_global:
            T = self._lidar_to_world(calib)
            ego_w = T @ np.array([ego_pose["x"], ego_pose["y"], ego_pose["z"], 1])
            Rlw = T[:3, :3]
            ego_world = ego_pose.copy()
            ego_world["x"], ego_world["y"], ego_world["z"] = ego_w[:3]
            ego_world["yaw"] = ego_pose["yaw"] + np.arctan2(Rlw[1, 0], Rlw[0, 0])
            cam_center = np.array([ego_world["x"], ego_world["y"], ego_world["z"]])
            self.update_ego_bbox(ego_world)
        else:
            cam_center = np.array([ego_pose["x"], ego_pose["y"], ego_pose["z"]])
            self.update_ego_bbox(ego_pose)

        # ------------------------------------------------ agent boxes
        dets = []
        for tr in tracklets:
            b = tr["current_pos"]
            dets.append(
                {
                    "label": tr["category"],
                    "x": b[0],
                    "y": b[1],
                    "z": b[2],
                    "dx": b[3],
                    "dy": b[4],
                    "dz": b[5],
                    "yaw": b[6],
                }
            )
        self.update_bboxes(dets)

        # trajectories
        self.update_trajectories(tracklets)

        # ------------------------------------------------ camera
        eye = cam_center + self.cam_offset
        force_camera_pose(self.vis, eye, cam_center, up=(0, 1, 0))
        self.vis.get_view_control().set_zoom(self.zoom)
        self.vis.poll_events(); self.vis.update_renderer()
