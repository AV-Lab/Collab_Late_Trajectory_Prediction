# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import networkx as nx
from collections import deque
from constants import Constants
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from utils import simulate_detection_error

class CollaborationGraph:
    
    
        # Constructor (optional)
    def __init__(self):
        # Initialize instance variables
        self.G = nx.Graph()
        
    class Node:
        def __init__(self, node_id, cur_location, last_updated, future_trajectory=None, pool=None, transmitting_node=False):
            self.node_id = node_id
            self.cur_location = cur_location
            self.past_trajectory = deque([np.zeros(6) for _ in range(Constants.MIN_TRACKING_FRAMES)], maxlen=Constants.MIN_TRACKING_FRAMES)
            self.future_trajectory = future_trajectory if future_trajectory is not None else []
            self.transmitting_node = transmitting_node
            self.last_updated = last_updated
            self.pool = pool if pool is not None else [] # prediction, timestamp, confidence, fps
        
        def add_observation_to_trajectory(self, observation, add_noise):
            distance = np.linalg.norm(observation)
            if add_noise:
                observation = simulate_detection_error(observation, distance, base_noise_std=0.5, distance_scaling_factor=0.001)
            self.past_trajectory.append(observation)

        def __repr__(self):
            return (f"Node(node_id={self.node_id}, "
                    f"cur_location={self.cur_location}, "
                    f"past_trajectory={list(self.past_trajectory)}, "
                    f"future_trajectory={self.future_trajectory}, "
                    f"transmitting_node={self.transmitting_node}, "
                    f"last_updated={self.last_updated})")


    def print_graph_size(self):
        print("Current size of the graph is :", len(self.G.nodes))
        
        
    def add_node(self, scene_id, node_id, state, cur_location, add_noise):        
        node = self.Node(node_id=node_id, cur_location=cur_location, last_updated=scene_id)
        node.add_observation_to_trajectory(state, add_noise)
        self.G.add_node(node_id, node_data=node)
        
        # Add edges with distances 
        for nid in self.G.nodes:
            if nid == node_id: continue
            location = self.G.nodes[nid]['node_data'].cur_location
            distance = ((location[0] - cur_location[0]) ** 2 + 
                            (location[1] - cur_location[1]) ** 2 + 
                                (location[2] - cur_location[2]) ** 2) ** 0.5    
    
            self.G.add_edge(node_id, nid, distance=distance)
                
    def update_node(self, scene_id, node_id, state, cur_location, add_noise):
        # Update nodes in the graph (in the dict, in the graph update location)
        node = self.G.nodes[node_id]['node_data']
        node.add_observation_to_trajectory(state, add_noise)
        node.last_updated = scene_id
        node.cur_location = cur_location
        self.G.nodes[node_id].update(node_data=node)
        
        # Add edges with distances (for simplicity, let's use Euclidean distance)
        for nid in self.G.nodes:
            if nid == node_id or not self.G.has_edge(node_id, nid): continue
            
            location = self.G.nodes[nid]['node_data'].cur_location
            distance = ((location[0] - cur_location[0]) ** 2 + 
                            (location[1] - cur_location[1]) ** 2 + 
                                (location[2] - cur_location[2]) ** 2) ** 0.5    
            
            self.G[node_id][nid].update(distance=distance)
    
    def update_observations(self, scene_id, observations, add_noise):
        # if the observation is in the graph just append the point
        # if observation is not in the graph add a node 
        for node_id, state in observations.items():
            cur_location = state[:3]
            if node_id in self.G.nodes:
                self.update_node(scene_id, node_id, state, cur_location, add_noise)
            else:
                self.add_node(scene_id, node_id, state, cur_location, add_noise)              
                
        # remove nodes, which were not associated for some time 
        current_nodes = copy.deepcopy(self.G.nodes)
        for nid in current_nodes:
            if abs(scene_id - self.G.nodes[nid]['node_data'].last_updated) > Constants.KEEP_TRACK: 
                self.G.remove_node(nid)
                
    def update_trajectories(self, objects_id, trajectories):
        for idx, obid in enumerate(objects_id):
            self.G.nodes[obid]['node_data'].future_trajectory = trajectories[idx].tolist()
    
    def associate(self, tmp, fps, ph, confidence, traj_e):
        min_dist = math.inf
        min_node_id = -1
        
        for nid in self.G.nodes:
            loc = self.G.nodes[nid]['node_data'].cur_location
            dist = np.linalg.norm(loc-traj_e[-1][:3]) 
            if dist < min_dist:
                min_dist = dist
                min_node_id = nid
        
        if min_dist > Constants.ASSC_THRESHOLD: return
        
        cur_confidence = np.linalg.norm(np.array(self.G.nodes[min_node_id]['node_data'].cur_location))
        
        #if confidence < cur_confidence:
        self.G.nodes[min_node_id]['node_data'].pool.append((tmp, fps, ph, confidence, traj_e[:-1]))
        
    
    def print_(self):
        for nid in self.G.nodes:
            print(self.G.nodes[nid]['node_data'])
            
    def get_tracks(self):
        return {nid: self.G.nodes[nid]['node_data'].past_trajectory for nid in self.G.nodes}
    
    def get_predictions(self):
        return [(self.G.nodes[nid]['node_data'].cur_location, self.G.nodes[nid]['node_data'].future_trajectory) for nid in self.G.nodes]
        
    
    def visualize_(self):
    # Create a plot
        plt.figure(figsize=(10, 8))
        
        # Draw the nodes and edges
        pos = nx.spring_layout(self.G)  # or use nx.spring_layout(G), nx.kamada_kaway_layout(G), etc.
        nx.draw_networkx_nodes(self.G, pos, node_size=500, node_color='lightblue', edgecolors='black')
        nx.draw_networkx_edges(self.G, pos, width=2, alpha=0.7, edge_color='gray')
        nx.draw_networkx_labels(self.G, pos, font_size=12, font_family='sans-serif')
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.G, 'distance')  # or any attribute you want to label
        formatted_edge_labels = {k: f"{v:.1f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=formatted_edge_labels, font_size=12, font_family='sans-serif')
    
        # Display the graph
        plt.title("Graph Visualization with Edge Labels")
        plt.show()
