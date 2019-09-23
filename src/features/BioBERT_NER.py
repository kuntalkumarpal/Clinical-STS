#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 19 Jun 2019

@author: samrawal
"""

import sys
sys.path.append("..")
from utils.biobert.generate_inference_file import bert_ner


def BioBERT_NER(sent1, additional_data=None):
  return bert_ner(sent1, cache=True)
