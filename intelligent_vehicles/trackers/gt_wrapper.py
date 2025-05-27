#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mar 16, 2025

@author: nadya

This class implements a simple ground-truth-based tracker. 
It maintains tracklets based on object IDs available in the ground-truth detections.

The `track()` method updates the internal list of active tracklets using the current frame's detections.
Each tracklet stores the object ID, category, confidence, current position, and a history of bounding boxes.

The `get_tracked_objects()` method returns a list of tracked objects, where each object is represented as:
    (ID, category, confidence, current_position, history)

- `ID`: unique object ID (int or string)
- `category`: object label/class (e.g., "car", "pedestrian")
- `confidence`: fixed at 1.0 (oracle tracker)
- `current_position`: instance of BBox class
- `history`: list of past BBox instances (up to `history_len`)
"""

import logging
from queue import Queue
import numpy as np 
from collections import namedtuple

logger = logging.getLogger(__name__)
position = namedtuple('Position', ['x', 'y', 'z','yaw'])

class GTWrapper:
    def __init__(self, history_len=20):
        logger.info("The tracklets are taken from ground truth.")
        self.active_tracklets = []
        self.history_len = history_len
        
    def wrap_angle(self, theta):
        if theta >= np.pi:
            return theta - 2 * np.pi
        elif theta < -np.pi:
            return theta + 2 * np.pi
        return theta

    def ego_motion_compensation(self, detections, calibration):
        """
        Convert LiDAR-frame boxes to world frame (position + yaw).
        """
    
        T_lw = calibration["ego_to_world"] @ calibration["lidar_to_ego"]  # 4×4
        R_lw = T_lw[:3, :3]
    
        # ego heading = yaw of LiDAR X-axis in world frame
        ego_heading = np.arctan2(R_lw[1, 0], R_lw[0, 0])   # atan2(y,x)
    
        compensated = []
        for det in detections:
            # ----- position
            pos_lidar = np.array([det["x"], det["y"], det["z"], 1.0])
            pos_world = T_lw @ pos_lidar
    
            # ----- yaw  (add headings, then wrap)
            yaw_world = det["yaw"] + ego_heading
            yaw_world = (yaw_world + np.pi) % (2 * np.pi) - np.pi   # wrap to (-π,π]
    
            new_det = det.copy()
            new_det["x"], new_det["y"], new_det["z"] = pos_world[:3]
            new_det["yaw"] = yaw_world
            compensated.append(new_det)
    
        return compensated


    def track(self, detections, ego_pose, calibartion):            
        transformed_detections = self.ego_motion_compensation(detections, calibartion)
        
        #transformed_detections = detections
        
        print("Detections:.........................................................................")
        for d in transformed_detections:
            print(d)
        
        unmatched_detections = []
        matched_track_ids = []
        

        for d in transformed_detections:
            matched = False
            for tr in self.active_tracklets:
                if d['obj_id'] == tr.id:
                    print("TRacklet before update")
                    print(tr)
                    tr.update(d)
                    print("TRacklet after update")
                    print(tr)
                    matched = True
                    matched_track_ids.append(tr.id)
                    break

            if not matched:
                unmatched_detections.append(d)

        # Keep only updated tracks
        matched_track_ids = set(matched_track_ids)
        self.active_tracklets = [tr for tr in self.active_tracklets if tr.id in matched_track_ids]

        # Initialize tracklets for unmatched detections
        for d in unmatched_detections:
            self.active_tracklets.append(self.Track(self.history_len, d))
            
            
        print("DONE tracking ................................................................................................................. \n\n\n\n\n")

    def get_tracked_objects(self):
        tracklets = [{'id' : tr.id, 'category' : tr.category, 'condifence' : tr.confidence,  
                 'current_pos' : tr.current_pos.to_array(), 'tracklet' : list(tr.history.queue)}
                for tr in self.active_tracklets]
        
        #print(tracklets)
        return tracklets 
    

    class Track:
        def __init__(self, len_history, detection):
            self.history = Queue(maxsize=len_history)
            self.active = True
            self.confidence = 1.0
            self.category = detection['label']
            self.id = detection['obj_id']
            self.current_pos = GTWrapper.BBox(detection)
            self.history.put(self.current_pos.to_position())

        def update(self, detection):
            cur_box = GTWrapper.BBox(detection)
            self.current_pos = cur_box
            if self.history.full():
                self.history.get()
            self.history.put(cur_box.to_position())
            self.active = True
            
        def __str__(self):
            return 'id: {}, category: {}, current_pos: {}, history: {}'.format(self.id, self.category, self.current_pos, list(self.history.queue))

    class BBox:
        def __init__(self, detection):
            self.x = detection['x']
            self.y = detection['y']
            self.z = detection['z']
            self.dx = detection['dx']
            self.dy = detection['dy']
            self.dz = detection['dz']
            self.yaw = detection['yaw']
            
        def to_array(self):
            return np.array([self.x, self.y, self.z, self.dx, self.dy, self.dz, self.yaw])
        
        def to_position(self):
            return position(self.x, self.y, self.z, self.yaw)