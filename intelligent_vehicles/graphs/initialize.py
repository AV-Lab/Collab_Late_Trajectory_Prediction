#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:54:00 2025

@author: nadya
"""

from intelligent_vehicles.graphs.object_graph import ObjectGraph
import logging

logger = logging.getLogger(__name__)

def initialize_object_graph():
    return ObjectGraph()