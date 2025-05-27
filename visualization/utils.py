#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:02:10 2025

@author: nadya
"""


import numpy as np
import open3d as o3d

def rotate_points_neg90_z(xyz):
    Rz_neg90 = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
    return (Rz_neg90 @ xyz.T).T

def rotate_box_neg90_z(center, yaw):
    x, y, z = center
    Rz_neg90 = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=float)
    rotated_center = Rz_neg90 @ np.array([x, y, z], dtype=float)
    rotated_yaw = yaw - np.pi / 2
    return rotated_center, rotated_yaw

def create_box_mesh_and_edges(center, length, width, height, yaw,
                              base_color, lighten_alpha=0.5):
    mesh = o3d.geometry.TriangleMesh.create_box(width=length, height=width, depth=height)
    mesh.translate(-np.array([length/2, width/2, height/2]))
    Rz = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    mesh.rotate(Rz, center=(0, 0, 0))
    mesh.translate(center)
    face_color = [base_color[i] + (1.0 - base_color[i]) * lighten_alpha for i in range(3)]
    mesh.paint_uniform_color(face_color)
    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    line_set.paint_uniform_color(base_color)
    return mesh, line_set

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
