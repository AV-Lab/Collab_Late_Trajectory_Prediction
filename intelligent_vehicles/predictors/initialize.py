#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:54:00 2025

@author: nadya
"""

from intelligent_vehicles.predictors.rnn_wrapper import RNNWrapper
from intelligent_vehicles.predictors.rnn_wrapper_nll import RNNWrapperNLL
from intelligent_vehicles.predictors.transformer_wrapper import TransformerWrapper

import logging

logger = logging.getLogger(__name__)


def initialize_predictor(predictor_config):
    if predictor_config["name"] == "lstm":
        return RNNWrapper(predictor_config)
    elif predictor_config["name"] == "lstm_nll":
        predictor_config["uncertainty_aware"] = True
        return RNNWrapperNLL(predictor_config)
    elif predictor_config["name"] == "transformer":
        return TransformerWrapper(predictor_config)
    else:
        logger.error("You specified unsupported predictor class in yaml.")
        exit
