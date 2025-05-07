#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 18:46:46 2025

@author: nadya
"""


class LateFuser:
    
    def __init__(self):
        pass
    
    def collab_predict(self, current_timestamp, tracks, trajectories):
        predictions = self.indiv_predict(tracks)
        fused_predictions = []
            
        for idx, key in enumerate(tracks.keys()):
            cur_confidence = np.linalg.norm(np.array(self.collaboration_graph.G.nodes[key]['node_data'].cur_location))
            shared_features = self.collaboration_graph.G.nodes[key]['node_data'].pool
            if len(shared_features) > 0:
                confidences = [sf[3] for sf in shared_features]
                shared_predictions = [sf[4][:,:3] for sf in shared_features]
                fused_prediction = self.predictor.fuse_late_weighted_average(cur_confidence, predictions[idx], confidences, shared_predictions)
                fused_predictions.append(fused_prediction)
                self.collaboration_graph.G.nodes[key]['node_data'].pool = [] 
            else:
                fused_predictions.append(predictions[idx])
                     
        return fused_predictions