#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:55:31 2025

@author: nadya
"""

from intelligent_vehicles.trackers.id_association_tracker import IDAssociationTracker
from intelligent_vehicles.trackers.metric_association_tracker import MetricAssociationTracker
from intelligent_vehicles.trackers.gt_tracker import GTTracker


def initialize_tracker(tracker_config):
    if tracker_config["name"] == "gt":
        return GTTracker(tracker_config["tracking_history"], tracker_config["observed_past"])
    if tracker_config["name"] == "id_association":
        return IDAssociationTracker(
            tracker_config["tracking_history"],
            tracker_config["fps"],
        )
    if tracker_config["name"] == "metric_association":
        return MetricAssociationTracker(
            tracker_config["tracking_history"],
            tracker_config["fps"],
        )
    else:
        print("You specified unsupported tracker class in yaml.")
        exit
