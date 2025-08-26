#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
World-frame visualiser (legacy Visualizer, open-top option + heading flags)

* Shows raw world-frame point cloud.
* Draws ego box (green) and agent boxes; roof triangles can be removed
  (open_top) so transparency works from above.
* Heading arrow:
      – always for the ego vehicle,
      – for agents only if `agent_heading=True`.
* Keeps API/attributes identical to the original version, so external code
  that relies on `bbox_geometries`, method names, etc. continues to work.
"""

from __future__ import annotations
import logging
from typing import Dict, List

import numpy as np
import open3d as o3d
from visualization.utils import create_box_mesh_and_edges, force_camera_pose

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# helper: cuboid with optional roof removal (unchanged behaviour)
# ────────────────────────────────────────────────────────────────────────────
def _create_box_mesh_and_edges_transparent(
    centre, l, w, h, yaw, colour, alpha=0.5, remove_top: bool = True
):
    mesh, edges = create_box_mesh_and_edges(centre, l, w, h, yaw, colour)

    mesh.paint_uniform_color(colour[:3])
    mesh.compute_vertex_normals()
    try:
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlitTransparency"
        mat.base_color = [*colour[:3], alpha]
        mat.transparency = alpha
        mesh.material = mat
    except Exception:
        pass

    if remove_top:
        verts = np.asarray(mesh.vertices)
        tris  = np.asarray(mesh.triangles)
        max_z = verts[:, 2].max() - 1e-6
        roof  = np.all(verts[tris][:, :, 2] > max_z, axis=1)
        mesh.remove_triangles_by_mask(roof)
        mesh.remove_unreferenced_vertices()

    return mesh, edges

# ────────────────────────────────────────────────────────────────────────────
# visualiser
# ────────────────────────────────────────────────────────────────────────────
class BBoxVisualizer:
    def __init__(
        self,
        camera_height: float = 40.0,
        zoom_level:   float = 0.15,
        open_top:     bool  = True,
        face_alpha:   float = 0.35,
        agent_heading: bool = False,          # ← NEW flag
    ):
        self.open_top       = open_top
        self.alpha          = face_alpha
        self.agent_heading  = agent_heading   # store flag

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("World-Frame Boxes", 1280, 720)

        opt = self.vis.get_render_option()
        opt.background_color      = np.array([0.1, 0.1, 0.1])
        opt.point_size            = 2.0
        opt.show_coordinate_frame = True

        self.cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.cloud)

        self.bbox_geometries: List[o3d.geometry.Geometry] = []
        self.label_colors: Dict[str, tuple] = {}

        self.cam_offset = np.array([0.0, 0.0, camera_height])
        self.zoom       = zoom_level

    # ........................................................ point cloud
    def update_cloud(self, pc_dict: Dict):
        xyz = pc_dict["data"][:, :3].astype(np.float32)
        self.cloud.points = o3d.utility.Vector3dVector(xyz)
        self.cloud.colors = o3d.utility.Vector3dVector(
            np.full_like(xyz, 1.0)
        )
        self.vis.update_geometry(self.cloud)

    # ........................................................ heading arrow
    def _add_heading(self, centre, length, yaw, colour):
        fwd   = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        nose  = np.array(centre) + fwd * (length / 2)
        tip   = nose + fwd * 1.5
        tip_len = 0.4
        ortho  = np.array([-fwd[1], fwd[0], 0])
        ortho  = ortho / (np.linalg.norm(ortho[:2]) + 1e-6) * (tip_len * 0.6)

        pts = [nose, tip, tip - fwd * tip_len + ortho, tip - fwd * tip_len - ortho]
        seg = [[0, 1], [2, 1], [3, 1]]

        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(seg),
        )
        ls.colors = o3d.utility.Vector3dVector([colour] * len(seg))
        self.vis.add_geometry(ls)
        self.bbox_geometries.append(ls)

    # ........................................................ one cuboid
    def _add_box(self, centre, l, w, h, yaw, colour):
        mesh, edges = _create_box_mesh_and_edges_transparent(
            centre, l, w, h, yaw, colour, self.alpha, self.open_top
        )
        self.vis.add_geometry(mesh)
        self.vis.add_geometry(edges)
        self.bbox_geometries.extend([mesh, edges])
        # note: no heading here – it’s added separately where needed

    # ........................................................ ego / agents
    def update_ego_bbox(self, ego_pose: Dict):
        if ego_pose is None:
            logger.error("No ego_pose provided")
            return
        self._add_box(
            (ego_pose["x"], ego_pose["y"], ego_pose["z"]),
            ego_pose["length"], ego_pose["width"], ego_pose["height"], ego_pose["yaw"],
            (0, 1, 0),
        )
        # ALWAYS draw heading for ego
        self._add_heading(
            (ego_pose["x"], ego_pose["y"], ego_pose["z"]),
            ego_pose["length"], ego_pose["yaw"],
            (0, 1, 0),
        )

    def update_bboxes(self, detections: List[Dict]):
        for d in detections:
            lbl = d.get("label", "obj")
            if lbl not in self.label_colors:
                self.label_colors[lbl] = tuple(np.random.rand(3))

            self._add_box(
                (d["x"], d["y"], d["z"]),
                d["dx"], d["dy"], d["dz"],
                d["yaw"],
                self.label_colors[lbl],
            )

            # Heading arrow for agents only if flag is True
            if self.agent_heading:
                self._add_heading(
                    (d["x"], d["y"], d["z"]),
                    d["dx"], d["yaw"],
                    self.label_colors[lbl],
                )

    # ........................................................ main render
    def visualize(self, pc_dict: Dict, detections: List[Dict], ego_pose: Dict):
        self.update_cloud(pc_dict)

        for g in self.bbox_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.bbox_geometries.clear()

        self.update_ego_bbox(ego_pose)
        self.update_bboxes(detections)

        centre = (
            np.array([ego_pose["x"], ego_pose["y"], ego_pose["z"]])
            if ego_pose else np.asarray(self.cloud.points).mean(axis=0)
        )
        eye = centre + self.cam_offset
        force_camera_pose(self.vis, eye, centre, up=(0, 1, 0))
        self.vis.get_view_control().set_zoom(self.zoom)

        self.vis.poll_events()
        self.vis.update_renderer()

    # ........................................................ cleanup
    def close(self):
        self.vis.destroy_window()
