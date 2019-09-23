#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gensim.models import KeyedVectors
#from gensim.test.utils import datapath
import pandas as pd
# from ast import literal_eval
import os


# In[2]:


### Set up root directory
print (os.getcwd())
os.chdir('homepath')
print (os.getcwd())


# In[3]:


df = pd.read_csv("src/data/allfeatures.csv",delimiter=",")


# In[4]:


### Download BIOWORDVEC if not present in data/embeddings



if not os.path.exists('data/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin'):
    raise ValueError("SKIP: You need to download the model")
    
wv = KeyedVectors.load_word2vec_format("data/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True)

def similarity(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1
    if len(a) == 0:
        return 0
    if len(b) == 0:
        return 0
    return wv.wmdistance(a, b)


# In[5]:



df['wordmover'] = df.apply(lambda x: similarity(x['Sequence1'], x['Sequence2']), axis=1)
#df['Domain_wordmover'] = df.apply(lambda x: similarity(x['Domain1'], x['Domain2']), axis=1)


# In[6]:


df.to_csv('src/data/allfeaturesWm.csv')


# In[7]:


df.head()


# In[ ]:




