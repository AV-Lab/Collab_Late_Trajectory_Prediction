#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
import open3d as o3d

from visualization.bbox_visualize import BBoxVisualizer
from visualization.utils import force_camera_pose

logger = logging.getLogger(__name__)


class PredictorVisualizer(BBoxVisualizer):
    """
    Trajectory + box visualiser.

    * If transform_to_global is True   →  LiDAR → world conversion applied
    * If transform_to_global is False →  data drawn in LiDAR frame
    """

    def __init__(self, camera_height=40.0, zoom_level=0.1):
        super().__init__(camera_height, zoom_level)
        self.trajectory_geometries = []

    # ------------------------------------------------------------------ traj
    def update_trajectories(self, tracklets, predictions, past_keep: int = 10):
        """
        Draw last `past_keep` history points in grey (small spheres) and the
        predicted future in the agent colour (large spheres).
        """
        grey = (0.5, 0.5, 0.5)          # RGB for past points
        for tr, pred in zip(tracklets, predictions):
            tracklet = tr["tracklet"]
            if len(tracklet) < 2:
                continue

            label = tr["category"]
            if label not in self.label_colors:
                self.label_colors[label] = np.random.rand(3)
            col = self.label_colors[label]

            # ---------------- past (keep only the last `past_keep` points)
            past   = tracklet[-past_keep:]
            past_z = past[-1].z                         # reference z for preds
            past_pts = np.array([(p.x, p.y, p.z) for p in past])

            # ---------------- future predictions (2-D or 3-D)
            fut_pts = []
            for v in pred.values():
                if len(v) == 2:                         # x,y  → pad z
                    fut_pts.append((v[0], v[1], past_z))
                else:
                    fut_pts.append(tuple(v))
            fut_pts = np.array(fut_pts)

            # ---------------- lines (past + future)
            all_pts = np.vstack([past_pts, fut_pts]) if len(fut_pts) else past_pts
            if len(all_pts) > 1:
                lines = [[i, i + 1] for i in range(len(all_pts) - 1)]
                ls = o3d.geometry.LineSet(
                    points=o3d.utility.Vector3dVector(all_pts),
                    lines=o3d.utility.Vector2iVector(lines),
                )
                # past segments grey, future segments agent colour
                ls.colors = o3d.utility.Vector3dVector(
                    [grey] * (len(past_pts) - 1) + [col] * (len(lines) - len(past_pts) + 1)
                )
                self.vis.add_geometry(ls)
                self.trajectory_geometries.append(ls)

            # ---------------- draw spheres
            for pt in past_pts:
                dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                dot.translate(np.asarray(pt) - np.array([0, 0, 0.025]))
                dot.paint_uniform_color(grey)
                self.vis.add_geometry(dot)
                self.trajectory_geometries.append(dot)

            for pt in fut_pts:
                dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
                dot.translate(np.asarray(pt) - np.array([0, 0, 0.1]))
                dot.paint_uniform_color(col)
                self.vis.add_geometry(dot)
                self.trajectory_geometries.append(dot)

    # ------------------------------------------------ helpers (unchanged)
    @staticmethod
    def _lidar_to_world(calib):
        return calib["ego_to_world"] @ calib["lidar_to_ego"]

    # ------------------------------------------------ cloud (unchanged)
    def update_cloud(self, pc_dict, calib, to_global: bool):
        xyz = pc_dict["data"][:, :3].astype(np.float32)

        if to_global:
            T = self._lidar_to_world(calib)
            xyz_h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
            xyz = (T @ xyz_h.T).T[:, :3]

        self.cloud.points = o3d.utility.Vector3dVector(xyz)
        self.cloud.colors = o3d.utility.Vector3dVector(
            np.full_like(xyz, 1.0)
        )
        self.vis.update_geometry(self.cloud)

    # ------------------------------------------------ main (unchanged logic)
    def visualize_predictions(
        self,
        point_cloud,
        tracklets,
        predictions,
        ego_pose,
        calib,
        transform_to_global: bool = True,
    ):
        self.update_cloud(point_cloud, calib, transform_to_global)

        # clear previous geometries
        for g in self.bbox_geometries + self.trajectory_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.bbox_geometries.clear()
        self.trajectory_geometries.clear()

        # ------------------ ego pose & boxes
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

        # ------------------ agent boxes
        dets = [
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
            for tr in tracklets
            for b in [tr["current_pos"]]
        ]
        self.update_bboxes(dets)

        # trajectories (NEW behaviour)
        self.update_trajectories(tracklets, predictions)

        # ------------------ camera
        eye = cam_center + self.cam_offset
        force_camera_pose(self.vis, eye, cam_center, up=(0, 1, 0))
        self.vis.get_view_control().set_zoom(self.zoom)
        self.vis.poll_events()
        self.vis.update_renderer()
