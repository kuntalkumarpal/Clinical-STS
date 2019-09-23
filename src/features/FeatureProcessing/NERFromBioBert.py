#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
from tqdm import tqdm


# In[2]:


### Set up root directory
print (os.getcwd())
os.chdir('homepath')
print (os.getcwd())


# ### 1. Source data processing

# In[3]:


import src.data.DataLoader as CustomDataLoader
from src.features.CustomTokenizer import CustomTokenizer


# In[4]:


pairs = CustomDataLoader.DataLoader()


# In[5]:


numOriginal = len(pairs)


# In[6]:


pairs[0]


# In[7]:



import sys
# sys.path.append("src/utils/biobert")
print (sys.path)
from src.features.BioBERT_NER import BioBERT_NER


# In[18]:


nerA = []
nerB = []
senta = []
sentb = []
score = []
for sent in tqdm(pairs):
    a,b = sent[0],sent[1]
#     print (a)
#     print (b)
    nera = BioBERT_NER(a)
    nerb = BioBERT_NER(b)
    nerA.append(nera[0])
    nerB.append(nerb[0])
    senta.append(a)
    sentb.append(b)
    score.append(sent[2].strip("\n"))
    #print(sent[2].strip("\n"))
#     print (nera[0])
#     print (nerb)
#     input("WAT")


# In[19]:


print(len(nerA),len(nerB))


# In[20]:


df = pd.DataFrame({'a':senta,'b':sentb,'score':score,'NERA':nerA,'NERB':nerB})


# In[21]:


df.head()


# In[22]:


df.to_csv("src/features/FeatureProcessing/NER.csv",index=False)


# In[ ]:




