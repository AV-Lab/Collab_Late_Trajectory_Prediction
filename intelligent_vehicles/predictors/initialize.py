#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:54:00 2025

@author: nadya
"""

from intelligent_vehicles.predictors.rnn_wrapper import RNNWrapper
import logging

logger = logging.getLogger(__name__)


def initialize_predictor(predictor_config):
    if predictor_config["name"] == "lstm":
        return RNNWrapper(predictor_config)
    else:
        logger.error("You specified unsupported predictor class in yaml.")
        exit
