#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 13:02:30 2025

@author: nadya
"""

import logging
import sys

def setup_logging(log_file):
    logger = logging.getLogger()  # root logger
    logger.setLevel(logging.DEBUG)  
    
    # 1. Console Handler: show INFO (and above) in console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  
    console_format = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_format)
    
    # 2. File Handler: include DEBUG (and above) in file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger