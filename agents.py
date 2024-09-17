#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:11:22 2024

@author: nadya
"""
from dataloaders.data_loader_deepaccident import DeepAccidentDataset
from visualization.visualize import Visualizer
from models.rnn_prediction import IndvidualRNNPredictor, MultiRNNPredictor
from collaboration import CollaborationGraph
import torch
import numpy as np

class IntelligentAgent:
    """ 
    Intelligent agent class.
    
    Parameters:
        name (str): Name of the agent.
        data_folder (str, optional): Folder for data.
        dataloader (object, optional): Dataloader object.
        predictor (object, optional): Predictor object.
        collaboration_graph (object, optional): Collaboration graph object.
    """
    
    def __init__(self, name, data_folder='', 
                             prediction_frequence=1, 
                             broadcasting_frequency=1, 
                             dataloader=None, 
                             predictor=None, 
                             collaboration_graph=None, 
                             prediction_horizon = 10,
                             fps = 10,
                             predictor_checkpoint_file=None):
        self.name = name
        self.prediction_frequence = prediction_frequence
        self.broadcasting_frequency = broadcasting_frequency
        self.prediction_horizon = prediction_horizon
        self.fps = fps
        self.data_folder = data_folder
        self.dataloader = dataloader
        self.predictor = predictor
        self.collaboration_graph = collaboration_graph
        self.location = None
        self.velocity = None
        self.size = None
        
        self._init_dataloader()
        self._init_predictor(predictor_checkpoint_file)
        self._init_collaboration_graph()
    
    def _init_dataloader(self):
        if self.dataloader is None:
            self.dataloader = DeepAccidentDataset(self.data_folder, agent=self.name)
    
    def _init_predictor(self, predictor_checkpoint_file):
        if self.predictor is None:
            self.predictor = MultiRNNPredictor(self.data_folder, agent=self.name, checkpoint_file=predictor_checkpoint_file)
    
    def _init_collaboration_graph(self):
        if self.collaboration_graph is None:
            self.collaboration_graph = CollaborationGraph()
            
    def get_observations(self, labels):
        return self.dataloader.extract_observations(labels)        
    
    def extract_current_tracklets(self):
        return self.collaboration_graph.get_tracks()   
    
    def query_ground_truth(self, scenario_id, scene_id, objects_id):
        return self.dataloader.query_ground_truth_trajerctory(scenario_id, scene_id, objects_id)
        
    def push_observations(self, scene_id, observations, add_noise=False):   
        if -100 in observations:
            state = observations[-100]
            self.location = state[:3]
            self.yaw = state[3]
            self.velocity = state[4:]
            del observations[-100]        
        self.collaboration_graph.update_observations(scene_id, observations, add_noise)
        
    def push_predictions(self, objects_id, trajectories):
        self.collaboration_graph.update_trajectories(objects_id, trajectories)
                
    def indiv_predict(self, tracks):
        list_from_deque = [[list(sub_deque) for sub_deque in q] for q in tracks.values()]
        tracklets = torch.tensor(list_from_deque, dtype=torch.float32)
        predictions = self.predictor.predict(tracklets)
        reshaped_predictions = [p.numpy()[:,:3] for p in predictions]
        return reshaped_predictions
    
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
    
    def transmit(self, scene_id, calib_matrix):
        """ Transform and broadcast"""
        
        if self.broadcasting_frequency != 1 and scene_id % self.broadcasting_frequency != 0:
            return None
        
        return self.create_package(scene_id, calib_matrix)
