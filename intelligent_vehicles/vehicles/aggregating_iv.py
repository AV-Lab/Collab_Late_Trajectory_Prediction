#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:11:22 2024

@author: nadya
"""

import torch
import numpy as np

class AggregatingIntelligentVehicle:
    """ 
    Intelligent agent class.
    
    Parameters:
        name (str): Name of the agent.
        data_folder (str, optional): Folder for data.
        dataloader (object, optional): Dataloader object.
        predictor (object, optional): Predictor object.
        collaboration_graph (object, optional): Collaboration graph object.
    """
    
    def __init__(self, name, detector, tracker, predictor, parameters, sensors, data):
        pass