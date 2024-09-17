#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 11:13:52 2024

@author: nadya
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
from constants import Constants
from evaluation import calculate_ade, calculate_fde
from utils import weighted_average, normalize_confidences


class IndvidualRNNPredictor:
        # Constructor (optional)
        
    class TrajectoryDataset(Dataset):
        def __init__(self, dataset_file):        
            with open(dataset_file, 'rb') as f:
                self.data = pickle.load(f)
                self.data = torch.tensor(self.data, dtype=torch.float32)
        
        def __len__(self):
            return len(self.data) 
        
        def __getitem__(self, idx):        
            return self.data[idx][0], self.data[idx][1]
        
        
    class Encoder(nn.Module):    
        def __init__(self, input_size, hidden_size):
            super().__init__()        
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        def forward(self, x):
            outputs, (hidden, cell) = self.lstm(x)        
            return hidden, cell
    
    
    # Define the Decoder
    class Decoder(nn.Module):    
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()        
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x, hidden, cell):        
            outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
            predictions = self.fc(outputs)        
            return predictions, hidden, cell
    
    # Define the Seq2Seq model
    class Seq2Seq(nn.Module):    
        def __init__(self, encoder, decoder):
            super().__init__()        
            self.encoder = encoder
            self.decoder = decoder
        
        def forward(self, source, target_len):
            batch_size = source.size(0) 
            hidden, cell = self.encoder(source)

            if batch_size == 1:
                hidden = hidden.squeeze(1)
                cell = cell.squeeze(1)
            # Prepare decoder input (initially zeros, then use previous prediction)
            decoder_input = torch.zeros(batch_size, 1, source.size(2)).to(source.device)        
            outputs = []
            
            for _ in range(target_len):
                decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)            
                outputs.append(decoder_output)
                decoder_input = decoder_output
                
            outputs = torch.cat(outputs, dim=1)
            return outputs

    def __init__(self, dataset_dir, agent, checkpoint_file=None):
        # Initialize instance variables

        #params
        self.hidden_size = 128
        self.input_size = 6
        self.output_size = 6  
        self.input_len = Constants.MIN_TRACKING_FRAMES
        self.target_len = Constants.PREDICTION_HORIZON
        self.num_epochs = 50
        self.batch_size = 2048 
        self.agent = agent

        # create datasets and loaders
        self.train_dataset = self.TrajectoryDataset(dataset_file = '{}/{}/{}_{}.pkl'.format(dataset_dir, 'train', agent, Constants.TRAJ_FILE_SUF))
        self.valid_dataset = self.TrajectoryDataset(dataset_file = '{}/{}/{}_{}.pkl'.format(dataset_dir, 'valid', agent, Constants.TRAJ_FILE_SUF))
        #self.test_dataset = self.TrajectoryDataset(dataset_file = '{}/{}/{}_{}.pkl'.format(dataset_dir, 'test', agent, Constants.TRAJ_FILE_SUF))
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        #self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        
        # create model
        encoder = self.Encoder(self.input_size, self.hidden_size)
        decoder = self.Decoder(self.input_size, self.hidden_size, self.output_size)
        self.model = self.Seq2Seq(encoder, decoder)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if checkpoint_file is not None:
            print('load weights from checkpoint')
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def train(self):
        print("Total batches: ", len(self.train_loader))
        for epoch in range(self.num_epochs):
            for inputs, targets in self.train_loader:        
                self.optimizer.zero_grad()
                outputs = self.model(inputs, self.target_len)        
                loss = self.criterion(outputs, targets)
                loss.backward()        
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
            
            
        print("Saving the checkpoint ....")
        self.checkpoint = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(self.checkpoint, '{}_trained_rnn_model.pth'.format(self.agent))
            
    def evaluate(self):
        av_ade = 0
        av_fde = 0
        for inputs, targets in self.valid_loader:
            outputs = self.predict(inputs)
            av_ade += calculate_ade(outputs, targets)
            av_fde += calculate_fde(outputs, targets)
        print("ADE: {}".format(av_ade/len(self.valid_loader)))
        print("FDE: {}".format(av_fde/len(self.valid_loader)))
        
    def predict(self, input_trajectory):
        predicted_trajectory = []
        with torch.no_grad():    
            predicted_trajectory = self.model(input_trajectory, self.target_len)
        return predicted_trajectory
    
    def fuse_late_weighted_average(self, cur_confidence, cur_prediction, confidences, shared_predictions):
        """
        Compute the weighted average of predictions including current predictions and confidences,
        normalizing the confidences before calculation.
    
        :param cur_confidence: Current confidence value for the current predictions.
        :param cur_predictions: List of current prediction values.
        :param confidences: List of confidence values for shared predictions.
        :param shared_predictions: List of lists of predictions corresponding to the confidences.
        :return: Weighted average of all predictions combined with current predictions.
        """
        
        # Normalize shared confidences
        confidences.append(cur_confidence)
        normalized_confidences = normalize_confidences(confidences)
        shared_predictions.append(cur_prediction)
        avg_combined_predictions = weighted_average(shared_predictions, normalized_confidences)
    
        return avg_combined_predictions
        
    

class MultiRNNPredictor:
    # Constructor (optional)
    
    class TrajectoryDataset(Dataset):
        def __init__(self, dataset_file):        
            with open(dataset_file, 'rb') as f:
                self.data = pickle.load(f)
                self.data = torch.tensor(self.data, dtype=torch.float32)
        
        def __len__(self):
            return len(self.data) 
        
        def __getitem__(self, idx):        
            return self.data[idx][0], self.data[idx][1]
        
        
    class Encoder(nn.Module):    
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()        
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        
        def forward(self, x):
            outputs, (hidden, cell) = self.lstm(x)        
            return hidden, cell
    
    
    # Define the Decoder
    class Decoder(nn.Module):    
        def __init__(self, input_size, hidden_size, output_size, num_layers):
            super().__init__()        
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x, hidden, cell):        
            outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
            predictions = self.fc(outputs)        
            return predictions, hidden, cell
    
    # Define the Seq2Seq model
    class Seq2Seq(nn.Module):    
        def __init__(self, encoder, decoder):
            super().__init__()        
            self.encoder = encoder
            self.decoder = decoder
        
        def forward(self, source, target_len):
            batch_size = source.size(0) 
            hidden, cell = self.encoder(source)

            if batch_size == 1:
                hidden = hidden.squeeze(1)
                cell = cell.squeeze(1)
            # Prepare decoder input (initially zeros, then use previous prediction)
            decoder_input = torch.zeros(batch_size, 1, source.size(2)).to(source.device)        
            outputs = []
            
            for _ in range(target_len):
                decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)            
                outputs.append(decoder_output)
                decoder_input = decoder_output
                
            outputs = torch.cat(outputs, dim=1)
            return outputs

    def __init__(self, dataset_dir, agent, num_layers=2, checkpoint_file=None):
        # Initialize instance variables

        #params
        self.hidden_size = 128
        self.input_size = 6
        self.output_size = 6  
        self.num_layers = 2  # Specify the number of RNN layers
        self.input_len = Constants.MIN_TRACKING_FRAMES
        self.target_len = Constants.PREDICTION_HORIZON
        self.num_epochs = 50
        self.batch_size = 2048 
        self.agent = agent

        # create datasets and loaders
        self.train_dataset = self.TrajectoryDataset(dataset_file = '{}/{}/{}_{}.pkl'.format(dataset_dir, 'train', agent, Constants.TRAJ_FILE_SUF))
        self.valid_dataset = self.TrajectoryDataset(dataset_file = '{}/{}/{}_{}.pkl'.format(dataset_dir, 'valid', agent, Constants.TRAJ_FILE_SUF))
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        # create model
        encoder = self.Encoder(self.input_size, self.hidden_size, num_layers=self.num_layers)
        decoder = self.Decoder(self.input_size, self.hidden_size, self.output_size, num_layers=self.num_layers)
        self.model = self.Seq2Seq(encoder, decoder)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        if checkpoint_file is not None:
            print('load weights from checkpoint')
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def train(self):
        print("Total batches: ", len(self.train_loader))
        for epoch in range(self.num_epochs):
            for inputs, targets in self.train_loader:        
                self.optimizer.zero_grad()
                outputs = self.model(inputs, self.target_len)        
                loss = self.criterion(outputs, targets)
                loss.backward()        
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
            
        print("Saving the checkpoint ....")
        self.checkpoint = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(self.checkpoint, '{}_trained_rnn2_model.pth'.format(self.agent))
            
    def evaluate(self):
        av_ade = 0
        av_fde = 0
        for inputs, targets in self.valid_loader:
            outputs = self.predict(inputs)
            av_ade += calculate_ade(outputs, targets)
            av_fde += calculate_fde(outputs, targets)
        print("ADE: {}".format(av_ade/len(self.valid_loader)))
        print("FDE: {}".format(av_fde/len(self.valid_loader)))
        
    def predict(self, input_trajectory):
        predicted_trajectory = []
        with torch.no_grad():    
            predicted_trajectory = self.model(input_trajectory, self.target_len)
        return predicted_trajectory
    
    def fuse_late_weighted_average(self, cur_confidence, cur_prediction, confidences, shared_predictions):
        # Normalize shared confidences
        confidences.append(cur_confidence)
        normalized_confidences = normalize_confidences(confidences)
        shared_predictions.append(cur_prediction)
        avg_combined_predictions = weighted_average(shared_predictions, normalized_confidences)
    
        return avg_combined_predictions
    

