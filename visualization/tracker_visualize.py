#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import open3d as o3d
import numpy as np
import time


def rotate_points_neg90_z(xyz):
    """
    Rotate Nx3 array 'xyz' by -90° around Z:
      (x, y, z) -> (y, -x, z)
    """
    Rz_neg90 = np.array([
        [ 0,  1, 0],
        [-1,  0, 0],
        [ 0,  0, 1]
    ], dtype=float)
    return (Rz_neg90 @ xyz.T).T

def rotate_box_neg90_z(center, yaw):
    """
    new_center = Rz_neg90 * old_center
    new_yaw = old_yaw - pi/2
    """
    x, y, z = center
    Rz_neg90 = np.array([
        [ 0,  1, 0],
        [-1,  0, 0],
        [ 0,  0, 1]
    ], dtype=float)
    rotated_center = Rz_neg90 @ np.array([x, y, z], dtype=float)
    rotated_yaw    = yaw - np.pi/2
    return rotated_center, rotated_yaw


def force_camera_pose(vis, eye, center, up=(0,1,0)):
    """
    Force the camera to look from 'eye' toward 'center'.
    We'll do a two-pass render outside so it doesn't get overridden.
    """
    view_ctl = vis.get_view_control()
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()

    eye = np.array(eye, dtype=float)
    center = np.array(center, dtype=float)
    up = np.array(up, dtype=float)

    forward = center - eye
    forward /= np.linalg.norm(forward)

    left = np.cross(up, forward)
    left /= np.linalg.norm(left)

    true_up = np.cross(forward, left)
    true_up /= np.linalg.norm(true_up)

    R = np.stack([left, true_up, forward], axis=1)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = eye

    cam_params.extrinsic = np.linalg.inv(extrinsic)
    view_ctl.convert_from_pinhole_camera_parameters(cam_params)

def create_box_mesh_and_edges(center, length, width, height, yaw,
                              base_color, lighten_alpha=0.5):
    """
    Triangular mesh for faces, plus line set for edges.
    Faces = lighter color, edges = base_color.
    """
    mesh = o3d.geometry.TriangleMesh.create_box(width=length, height=width, depth=height)
    mesh.translate(-np.array([length/2, width/2, height/2]))

    Rz = o3d.geometry.get_rotation_matrix_from_xyz((0,0,yaw))
    mesh.rotate(Rz, center=(0,0,0))
    mesh.translate(center)

    face_color = [base_color[i] + (1.0 - base_color[i]) * lighten_alpha for i in range(3)]
    mesh.paint_uniform_color(face_color)

    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    line_set.paint_uniform_color(base_color)
    return mesh, line_set


class BBoxVisualizer:
    """
    1) We rotate entire scene by -π/2 => top->bottom -> bottom->top
    2) We have a top-down camera ignoring yaw of ego
    3) Ego bounding box is green, plus black 3D text label "EGO"
    4) Others => consistent color per label, plus black 3D text label
    5) zoom via set_zoom
    """
    def __init__(self, camera_height=30.0, zoom_level=0.15):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Ego BBox + Others + Black Text", width=1280, height=720)

        render_opt = self.vis.get_render_option()
        render_opt.background_color = np.array([0.1, 0.1, 0.1])
        render_opt.point_size = 2.0
        render_opt.show_coordinate_frame = True

        self.cloud = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.cloud)

        self.bbox_geometries = []

        self.cam_eye_local  = np.array([0.0, 0.0, camera_height])
        self.cam_look_local = np.array([0.0, 0.0, 0.0])
        self.zoom_level     = zoom_level
        
        self.label_colors = {}


    def update_cloud(self, points_dict):
        xyz = points_dict["data"][:, :3].astype(np.float32)
        xyz = rotate_points_neg90_z(xyz)
        self.cloud.points = o3d.utility.Vector3dVector(xyz)
        colors = np.tile([[1,1,1]], (xyz.shape[0],1))
        self.cloud.colors = o3d.utility.Vector3dVector(colors)
        self.vis.update_geometry(self.cloud)
        
    def update_ego_bbox(self, ego_pose):
        x, y, z = ego_pose['x'], ego_pose['y'], ego_pose['z']
        yaw     = ego_pose['yaw']
        w       = ego_pose['width']
        h       = ego_pose['height']
        l       = ego_pose['length']

        new_center, new_yaw = rotate_box_neg90_z((x,y,z), yaw)
        ego_color = (0.0, 1.0, 0.0)
        mesh, lines = create_box_mesh_and_edges(
            center=new_center,
            length=l,
            width=w,
            height=h,
            yaw=new_yaw,
            base_color=ego_color
        )
        self.vis.add_geometry(mesh)
        self.vis.add_geometry(lines)
        self.bbox_geometries.append(mesh)
        self.bbox_geometries.append(lines)

    def update_bboxes(self, detections):
        for det in detections:
            label  = det.get('label', 'obj')
            center = det['position']
            yaw    = det['yaw']
            length = det['length']
            width  = det['width']
            height = det['height']

            # rotate by -π/2
            new_center, new_yaw = rotate_box_neg90_z(center, yaw)

            # consistent color for each label
            if label not in self.label_colors:
                self.label_colors[label] = (np.random.rand(), np.random.rand(), np.random.rand())
            base_color = self.label_colors[label]

            mesh, lines = create_box_mesh_and_edges(
                center=new_center,
                length=length,
                width=width,
                height=height,
                yaw=new_yaw,
                base_color=base_color
            )
            self.vis.add_geometry(mesh)
            self.vis.add_geometry(lines)
            self.bbox_geometries.append(mesh)
            self.bbox_geometries.append(lines)

    def visualize(self, point_cloud, detections, ego_pose):
        """
        1) Rotate entire scene (cloud + boxes) by -π/2
        2) top-down ignoring yaw
        3) ego box in green
        4) others => consistent label color
        5) set_zoom
        """
        self.update_cloud(point_cloud)

        # remove old bounding boxes
        for g in self.bbox_geometries:
            self.vis.remove_geometry(g, reset_bounding_box=False)
        self.bbox_geometries.clear()

        self.update_ego_bbox(ego_pose)
        self.update_bboxes(detections)

        self.vis.poll_events()
        self.vis.update_renderer()

        # camera
        ex, ey, ez = ego_pose['x'], ego_pose['y'], ego_pose['z']
        cam_eye_world  = self.cam_eye_local  + np.array([ex, ey, ez])
        cam_look_world = self.cam_look_local + np.array([ex, ey, ez])
        force_camera_pose(self.vis, cam_eye_world, cam_look_world, up=(0,1,0))
        view_ctl = self.vis.get_view_control()
        view_ctl.set_zoom(self.zoom_level)

        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()
