#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World-frame visualiser (no rotations)
------------------------------------
• Shows raw (already-world) point cloud as-is.
• Draws ego box (green) and agent boxes.
• Adds a 2-D heading line in front of each box (derived from yaw).
"""

import logging
from typing import Dict, List
import numpy as np
import open3d as o3d

from visualization.utils import create_box_mesh_and_edges, force_camera_pose

logger = logging.getLogger(__name__)


class BBoxVisualizer:
    def __init__(self, camera_height: float = 60.0, zoom_level: float = 0.15):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("World-Frame Boxes", 1280, 720)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.1, 0.1, 0.1])
        opt.point_size = 2.0
        opt.show_coordinate_frame = True

        self.cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.cloud)

        self.bbox_geometries: List[o3d.geometry.Geometry] = []
        self.label_colors: Dict[str, tuple] = {}

        self.cam_offset = np.array([0.0, 0.0, camera_height])
        self.zoom = zoom_level

    # ------------------------------------------------------------- point cloud
    def update_cloud(self, pc_dict: Dict):
        xyz_world = pc_dict["data"][:, :3].astype(np.float32)
        self.cloud.points = o3d.utility.Vector3dVector(xyz_world)
        self.cloud.colors = o3d.utility.Vector3dVector(
            np.tile([[1, 1, 1]], (xyz_world.shape[0], 1))
        )
        self.vis.update_geometry(self.cloud)

    # ------------------------------------------------------------- boxes + heading
    def _add_box(self, center, l, w, h, yaw, color):
        # ------------------ main cuboid
        mesh, edges = create_box_mesh_and_edges(center, l, w, h, yaw, color)
        self.vis.add_geometry(mesh)
        self.vis.add_geometry(edges)
        self.bbox_geometries += [mesh, edges]
    
        # ------------------ heading arrow (all Lines, no rotations)
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])          # forward dir
        start   = np.array(center) + heading * (l / 2)               # box nose
        end     = start + heading * 1.5                              # 1.5 m ahead
        tip_len = 0.4                                                # tip size
        tip_off = heading * tip_len
    
        # orthogonal vector in XY plane for the V-tip
        ortho = np.array([-heading[1], heading[0], 0.0])
        ortho = ortho / (np.linalg.norm(ortho) + 1e-6) * (tip_len * 0.6)
    
        left  = end - tip_off + ortho
        right = end - tip_off - ortho
    
        pts  = [start, end, left, right]
        segs = [[0,1], [2,1], [3,1]]            # shaft + two tip lines
        arrow_ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(segs)
        )
        arrow_ls.colors = o3d.utility.Vector3dVector([color] * len(segs))
    
        self.vis.add_geometry(arrow_ls)
        self.bbox_geometries.append(arrow_ls)


    def update_ego_bbox(self, ego_pose: Dict):
        if ego_pose is None:
            logger.error("No ego_pose for this frame")
            return
        self._add_box(
            (ego_pose["x"], ego_pose["y"], ego_pose["z"]),
            ego_pose["length"],
            ego_pose["width"],
            ego_pose["height"],
            ego_pose["yaw"],
            (0, 1, 0),
        )

    def update_bboxes(self, detections: List[Dict]):
        for d in detections:
            lbl = d.get("label", "obj")
            if lbl not in self.label_colors:
                self.label_colors[lbl] = (
                    np.random.rand(),
                    np.random.rand(),
                    np.random.rand(),
                )
            self._add_box(
                (d["x"], d["y"], d["z"]),
                d["dx"],
                d["dy"],
                d["dz"],
                d["yaw"],
                self.label_colors[lbl],
            )

    # ------------------------------------------------------------- main render
    def visualize(self, pc_dict: Dict, detections: List[Dict], ego_pose: Dict):
        self.update_cloud(pc_dict)

        # clear old
        for g in self.bbox_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.bbox_geometries.clear()

        # draw boxes + heading lines
        self.update_ego_bbox(ego_pose)
        self.update_bboxes(detections)

        # camera
        if ego_pose is not None:
            center = np.array([ego_pose["x"], ego_pose["y"], ego_pose["z"]])
        else:
            pts = np.asarray(self.cloud.points)
            center = pts.mean(axis=0) if pts.size else np.zeros(3)
        eye = center + self.cam_offset
        force_camera_pose(self.vis, eye, center, up=(0, 1, 0))
        self.vis.get_view_control().set_zoom(self.zoom)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()
