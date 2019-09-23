#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gensim.models as word2vec
import pandas as pd
import os
from tqdm import tqdm
import time
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
import pickle


# In[2]:


print (os.getcwd())
os.chdir('homepath')
print (os.getcwd())


# In[3]:


embedding = "data/embeddings/pubmed_s100w10_min.bin"
data = "srcdata"


# In[4]:



# #Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
# #model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)


import gensim
model = word2vec.KeyedVectors.load_word2vec_format(embedding, binary=True)

# # Deal with an out of dictionary word: Михаил (Michail)
# if 'cholangiocarcinoma' in model:
#     print(model['cholangiocarcinoma'].shape)
# else:
#     print('{0} is an out of dictionary word'.format('cholangiocarcinoma'))
    
# print(model.most_similar(positive=['woman', 'king'], negative=['man']))

# print(model.doesnt_match("breakfast cereal dinner lunch".split()))

# print(model.similarity('woman', 'man'))

# print(model.most_similar('cholangiocarcinoma'))
# print(model.most_similar('arimidex'))


# In[5]:


print (model['2'].shape)


# In[7]:


print(model.most_similar('arimidex'))


# In[24]:


print(model.similarity('cholecalciferol', 'multivitamins'))


# In[ ]:




