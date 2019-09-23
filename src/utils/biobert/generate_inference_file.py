#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, pickle, sqlite3, os, csv
from collections import defaultdict
from tqdm import tqdm
sys.path.append("../..")
from data.DataLoader import DataLoader
from features.CustomTokenizer import CustomTokenizer

abspath = os.path.dirname(os.path.abspath(__file__))

def get_augmented_data():
  aug_data = []
  with open('srcdata', 'r') as aug:
    csv_reader = csv.reader(aug, delimiter=',')
    for row in csv_reader: aug_data.append((row[0], row[1], row[2]))
  return aug_data[1:] # remove out header

def get_test_data():
  test_data = []
  with open('srcdata2', 'r') as test:
    for line in test:
      split = line.split('\t')
      test_data.append((split[0],split[1],-1))
  return test_data
      
def generate_inference_file(filename='test.tsv', cache=True):
  if cache:
    return pickle.load(open(f'{abspath}/ser/sents.ser','rb'))
  else:
    sents = []
    tokenized_sents = []
    pairs = DataLoader('srcdata2')
    pairs = pairs + get_augmented_data() + get_test_data()
    for pair in tqdm(pairs, desc='Tokenizing sentences'):
      sents.append(pair[0])
      tokenized_sents.append(CustomTokenizer(pair[0]))
      sents.append(pair[1])
      tokenized_sents.append(CustomTokenizer(pair[1]))
      pickle.dump(sents, open(f'{abspath}/ser/sents.ser','wb'))
    inf_file = open(filename, 'w')
    for sent in tqdm(tokenized_sents, desc='Writing tokenized sentences'):
      for token in sent:
        inf_file.write(f'{token.strip()}\tO\n')
      inf_file.write('\n')
    return sents
        

def parse_bert_ner(bert_output='bert_NER_results.txt', cache=True):
  if cache:
    return pickle.load(open(f'{abspath}/ser/bert_ner.ser','rb'))
  else:
    bert_ner = []
    sent_ner = []
    bert = open(bert_output, 'r').readlines()
    for line in tqdm(bert, desc="Parsing BERT NER"):
      if line == '\n': # break signifiying new sentence
        bert_ner.append(sent_ner)
        sent_ner = []
      else:
        split = line.strip().split(' ')
        sent_ner.append((' '.join(split[:-2]), split[-1])) #(token, NE)
    pickle.dump(bert_ner, open(f'{abspath}/ser/bert_ner.ser','wb'))
    return bert_ner

def bert_ner(sentence, cache=True):
  if cache:
    sent2ner = pickle.load(open(f'{abspath}/ser/sent2ner.ser', 'rb'))
  else:
    sent2ner = defaultdict(list)
    sents = pickle.load(open(f'{abspath}/ser/sents.ser','rb'))
    bert_ner = pickle.load(open(f'{abspath}/ser/bert_ner.ser','rb'))

    if len(sents) != len(bert_ner):
      raise "Error -- len(sents) != len(NER)"
    else:
      for sent, ner in tqdm(zip(sents, bert_ner), desc="Serializing sent2ner"):
        sent2ner[sent].append(ner)
    pickle.dump(sent2ner, open(f'{abspath}/ser/sent2ner.ser', 'wb'))

  if sentence is not None and sentence in sent2ner:
    return sent2ner[sentence]
  else:
    clean_sent = sentence.replace(r'"', '')
    for sent in sent2ner:
      if sent.replace(r'"', '') == clean_sent:
        return sent2ner[sent]
    for sent in sent2ner:
      if sent.replace(r'"', '').strip() == clean_sent.strip():
        return sent2ner[sent]
    return -1


  
if __name__ == '__main__':
  #generate_inference_file(cache=False)
  parse_bert_ner(cache=False)
  bert_ner(None, cache=False)
