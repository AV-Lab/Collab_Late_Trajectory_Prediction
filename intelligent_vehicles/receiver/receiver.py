#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:36:01 2025

@author: nadya
"""

import redis
import json

class Receiver:
    # get from channel 
    # transform to local ccordinates  
    # run collaborator associate 
    
    def listen_for_predictions(vehicle_id):
        r = redis.Redis(host='localhost', port=6379)
        pubsub = r.pubsub()
        channel_name = f"predictions:{vehicle_id}"
        pubsub.subscribe(channel_name)
    
        print(f"Subscribed to {channel_name} ...")
        for msg in pubsub.listen():
            if msg["type"] == "message":
                data = json.loads(msg["data"])
                print(f"Received predictions for vehicle {vehicle_id} -> {data}")
                # do something with 'data'
                
    def associate(self, transmitted_package, calib_matrix):
        """ For each other vehicle tramsform the package, associate, and add to the pool"""
        if transmitted_package is None: return
        
        tmp = transmitted_package['timestamp'] 
        loc = transmitted_package['header'][0]
        fps = transmitted_package['header'][1]
        ph = transmitted_package['header'][2]
        inverse_W = np.linalg.inv(calib_matrix['ego_to_world'])
        inverse_E = np.linalg.inv(calib_matrix['lidar_to_ego'])
        
        for confidence, prediction in transmitted_package['body']:
            traj_w = np.dot(inverse_W, prediction.T).T
            traj_e = np.dot(inverse_E, traj_w.T).T
            self.collaboration_graph.associate(tmp, fps, ph, confidence, traj_e)