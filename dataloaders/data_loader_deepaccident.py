#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 23:17:53 2024

@author: nadya
"""

#import torch
#from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import cv2
import pickle
import torch
from queue import Queue
from utils import preprocess_dataset, generate_train_samples, parse_state, extract_trajectories_per_scene
from constants import Constants
import copy


class DeepAccidentDataset():
    """Dataset class for DeepAccident data."""
    
    
    def __init__(self, dataset_dir, agent):
        """
        Args:
            root_dir (string): Directory with all the sensors data.
        """
        self.agent = agent
        self.dataset_dir = dataset_dir
        self.objects_per_scenes = {} # pool of objects per scene 
        self.object_trajectories = {} # pool of gt trajectories per object
        
        # Load train dataset and extract trajectories 
        #self.load_dataset('train')
        #self.exctract_trajectories()
        
        # Load valid dataset and extract trajectories 
        self.load_dataset('valid')
        self.exctract_trajectories()
        
        # Load test dataset and extract trajectories 
        #self.load_dataset('test')
        #self.exctract_trajectories()

                      
    def load_dataset(self, prefix):
        # LOAD PICKLE
        with open('{}/{}/{}_{}.pkl'.format(self.dataset_dir, prefix, self.agent, Constants.SENSORS_DATA_SUF), 'rb') as f:
            self.data = pickle.load(f)
        
        avg_scenes = np.mean([len(v) for (k,v) in self.data.items()])
        
        print('{}: {}'.format("The total number of scenarios", len(self.data)))
        print('{}: {}'.format("Average number of scenes per scenario", int(avg_scenes)))
        
        self.keys = list(self.data.keys())
        
    def load_train(self):
        self.load_dataset('train')
    
    def load_test(self):
        self.load_dataset('test')
    
    def load_valid(self):
        self.load_dataset('valid')
    
    def __len__(self):
        return len(self.data)
    
    def extract_observations(self, scene_lables):
        # Extracts observations from the current frame
        observations = {}
        with open(scene_lables, 'r') as file:
            next(file)
            for line in file:
                data = line.strip().split(' ')
                id_ = int(data[10])
                if id_ >= 20000: continue             
                label, width, height, length, vector = parse_state(data)
                observations[id_] = vector
                
        return observations
        
    
    def exctract_trajectories(self):
        for scenario, sample in self.data.items():
            for scene_id, scene_data in sample.items(): # iterate through scenes in scenario
                sid = scenario + '_' + scene_id
                self.objects_per_scenes[sid] = []
                with open(scene_data['labels'], 'r') as file:
                    next(file)
                    for line in file:
                        data = line.strip().split(' ')
                        id_ = int(data[10])
                        if id_ < 0 or id_ >= 20000:
                            continue             
                        label, width, height, length, vector = parse_state(data)
                        oid = scenario + '_' + str(id_)
                        if oid in self.object_trajectories:
                            self.object_trajectories[oid][1].append(vector[:3])
                        else:
                            self.object_trajectories[oid] = (scene_id, [vector[:3]])     
                        self.objects_per_scenes[sid].append(oid)
                
    def query_ground_truth_trajerctory(self, scenario_id, scene_id, oids):
        scene_objects = self.objects_per_scenes[scenario_id + '_' + scene_id]
        trajectories = extract_trajectories_per_scene(scene_id, self.object_trajectories, scene_objects)
        scene_oids = {int(scene_obj.split('_')[-1]): idx for idx, scene_obj in enumerate(scene_objects)}
        gt_trajectories = [trajectories[scene_oids[oid]] if oid in scene_oids else []  for oid in oids]      
        return gt_trajectories

    def __getitem__(self, idx):
        # Retrieve and return a single sample from the dataset at the given index
        
        if self.data == None:
            print("Call function load_train, load_valid or load_test to load dataset first")
            exit()
        
        sample = self.data[self.keys[idx]]
        loaded_sample = {}
        
        for scene_id, scene_data in sample.items(): # iterate through scenes in scenario
            with open(scene_data['calib'], 'rb') as f:
                calib_matrix = pickle.load(f)
            loaded_sample[scene_id] = {'images': [], 'labels': scene_data['labels'], 'calib': calib_matrix}
            #images_data = []
            #for image_path in scene_data['images']:
            #    images_data.append(cv2.imread(image_path))
            #loaded_sample[scene_id]['images'] =  images_data

            
        return self.keys[idx], loaded_sample
     
    
