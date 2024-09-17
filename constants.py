#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:55:41 2024

@author: nadya
"""


class Constants:
    MIN_TRACKING_FRAMES = 10
    SLIDING_WINDOW = 1
    PREDICTION_HORIZON = 10
#    SENSORS = ['Camera_FrontLeft', 
#               'Camera_Front',
#               'Camera_FrontRight', 
#               'Camera_BackLeft', 
#               'Camera_Back', 
#               'Camera_BackRight']
    SENSORS=['Camera_Front']
    ANN_DIR = 'label'
    CALLIB_DIR = 'calib'
    AGENTS = ['ego_vehicle', 
              'ego_vehicle_behind', 
              'other_vehicle', 
              'other_vehicle_behind']
    TRAJ_FILE_SUF = 'trajectories_data'
    SENSORS_DATA_SUF = 'sensors_data'
    KEEP_TRACK = 5
    THRESHOLD_DIST = 30
    ASSC_THRESHOLD = 0.5