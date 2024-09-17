#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:38:28 2024

@author: nadya
"""
import cv2
import matplotlib
import pickle
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from utils import Constants
import numpy as np


inverse_transformation = np.array([[0,1,0],[0,0,-1],[1,0,0]])

class Visualizer:
    
    def __init__(self, output_folder):
        self.output_folder = output_folder
    
    def project_lidar_to_image(self, T, K, lidar_points_homogeneous):
        points_camera_coords = np.dot(T, lidar_points_homogeneous.T).T
        points_camera_coords = points_camera_coords[:, :3]
        projected_points = np.dot(K, points_camera_coords.T).T
        projected_points[:,0] /= projected_points[:,2]
        projected_points[:,1] /= projected_points[:,2]
        return projected_points

    
    def transform_trajectories(self, image, trajectories, sensor, calib_matrix):
        intrisic_matrix = np.array(calib_matrix['intrinsic_{}'.format(sensor)])
        transformation_matrix = np.array(calib_matrix['lidar_to_{}'.format(sensor)])
        img_height, img_width, _ = image.shape
        
        for trajectory in trajectories:
            trajectory_homog = np.array([t + [1] for t in trajectory])
            projected_points = self.project_lidar_to_image(transformation_matrix, intrisic_matrix, trajectory_homog)
            c1 = projected_points[:,0] > 0 
            c2 = projected_points[:,0] < img_width
            c3 = projected_points[:,1] > 0 
            c4 = projected_points[:,1] < img_height
            c5 = projected_points[:,2] > 0.5
            filtered_projected_points = projected_points[c1&c2&c3&c4&c5]
            if len(filtered_projected_points) > 2:
                start_point = (int(filtered_projected_points[0][0]), int(filtered_projected_points[0][1]))
                end_point = (int(filtered_projected_points[-1][0]), int(filtered_projected_points[-1][1]))
                print()
                cv2.arrowedLine(image, start_point, end_point, color=(255, 0, 0), thickness=3, tipLength=0.1)
            #for x,y,z in projected_points:
            #    if z > 0 and 0 < x < img_width and 0 < y < img_height:
            #        cv2.circle(image, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)
        return image
    
    def extract_trajectories_per_scene(self, scene_id, objects_trajectories, objects_per_scene):
        current_objects = objects_per_scene[scene_id]
        trajectories = []
        for obj in current_objects:
            bgs = int(objects_trajectories[obj][0])
            sid = int(scene_id)
            lb = sid - bgs
            rb = lb + Constants.PREDICTION_HORIZON
            traj = objects_trajectories[obj][1]
            if lb > 0 and rb < len(traj): 
                trajectories.append(traj[lb:rb])
                
        return trajectories

    
    def create_multi_view(self, images, calib_matrix, trajectories, layout=(1, 1)):
        rows, cols = layout
        img_height, img_width, _ = images[0].shape
        multi_view = np.zeros((img_height * rows, img_width * cols, 3), dtype=np.uint8)
        for i, img in enumerate(images):
            r = i // cols
            c = i % cols
            if len(trajectories) > 0:
                img = self.transform_trajectories(img, trajectories, Constants.SENSORS[i], calib_matrix)
                #plt.imshow(img)
                #plt.show()
            multi_view[r*img_height:(r+1)*img_height, c*img_width:(c+1)*img_width, :] = img
        return multi_view
    
    def visualize_scenario(self, scenario_id, scenario_data, objects_trajectories=None, objects_per_scene=None):
        
        output_video_path = '{}\{}.mp4'.format(self.output_folder, scenario_id)
        frame_rate = 7
        
        with_annotations = False if objects_trajectories is None else True

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (1600, 900)) # 4800, 1800
        for scene_id, scene_data in scenario_data.items():            
            trajectories = []
            if with_annotations:
                trajectories = self.extract_trajectories_per_scene(scene_id, objects_trajectories, objects_per_scene)
            multi_view_image = self.create_multi_view(scene_data['images'], scene_data['calib_matrix'], trajectories)              
            out.write(multi_view_image)
        out.release()
        print(f"Video saved to {output_video_path}")
        


            
