#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:33:23 2019

@author: ishanshrivastava
"""

from data.DataLoader import DataLoader
#from features.CustomTokenizer import CustomTokenizer

pairs = DataLoader("../../data/train/jsonl/clinicalSTS2019_dev.jsonl")
#print(CustomTokenizer(pairs[3][0]))