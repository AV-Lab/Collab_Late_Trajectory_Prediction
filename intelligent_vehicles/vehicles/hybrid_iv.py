#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:11:22 2024

@author: nadya
"""

import torch
import numpy as np

class HybridIV:
    """ 
    Intelligent agent class.
    
    Parameters:
        name (str): Name of the agent.
        data_folder (str, optional): Folder for data.
        dataloader (object, optional): Dataloader object.
        predictor (object, optional): Predictor object.
        prediction_map (object, optional): Agent-level prediction map.
    """
    

    def __init__(self, 
                 name, 
                 detector_config, 
                 tracker_config, 
                 predictor_config, 
                 broadcaster_config, 
                 listener_config, 
                 parameters, 
                 sensors, 
                 data, 
                 clock_step, 
                 channel_root,
                 load_lidar):
        pass
    
