#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:39:22 2025

@author: nadya
"""

import redis
import json


class Broadcaster:
    # transform to global coordinates 
    # create a package
    # wirte to the channel
    
    def __init__():
        pass
    
    def transform_to_global(self):
        pass 
    
    def create_package(self, current_timestamp, calib_matrix):
    """ Package structure:
        timestamp, my_location, my_parameters 
        for each agent current_location, prediction
    """
    
    E = calib_matrix['lidar_to_ego']
    W = calib_matrix['ego_to_world']
    predictions = self.collaboration_graph.get_predictions()
    package = {}
    package['timestamp'] = current_timestamp
    package['header'] = (self.location, self.fps, self.prediction_horizon)
    package['body'] = []
    for p in predictions:
        t = np.array(p[1])[:,:3]
        t = np.vstack([t, p[0]])
        t = np.append(t, np.ones((len(t), 1)), axis=1)
        t_ego = np.dot(E, t.T).T
        t_world = np.dot(W, t_ego.T).T
        confidence = np.linalg.norm(np.array(p[0]))
        package['body'].append((confidence, t_world))
    return package

    def broadcast_predictions(vehicle_id, predictions):
        r = redis.Redis(host='localhost', port=6379)
        channel_name = f"predictions:{vehicle_id}"
        message = json.dumps(predictions)  # convert data to JSON
        r.publish(channel_name, message)