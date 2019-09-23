#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import ast

def read_jsonl(filename):
  json_file = open(filename)
  json_str = json_file.readlines()
  pairs = list()
  for s in json_str:
    oneDict = ast.literal_eval(s)
    if oneDict['gold_label'] == '-':
      continue
    pairs.append((oneDict['sentence1'], oneDict['sentence2'], oneDict['given_gold_label']))
  return pairs 

def read_sts(filename):
    lines = open(filename,encoding="utf-8").readlines()
    pairs = list()
    for s in lines:
        split = s.split('\t')
        if split is not None:
            if len(split)==3:
                pairs.append(tuple((split[0],split[1],split[2])))
            else:
                pairs.append(tuple((split[0],split[1])))
    return pairs

    

def DataLoader(tsvPath="<srcdata>", parameters=None):
    #base_path=os.path.dirname(os.path.abspath(__file__))
    base_path=""
    if tsvPath.split('.')[-1].split(" ")[0] in ['txt','tsv']:
        return read_sts(os.path.join(base_path,tsvPath))
    elif 'jsonl' in tsvPath.split('.')[-1]:
        return read_jsonl(os.path.join(base_path,tsvPath))
    else:
        raise ValueError('Invalid file. File should be either a txt or jsonl')
