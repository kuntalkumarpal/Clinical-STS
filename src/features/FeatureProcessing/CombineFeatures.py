#!/usr/bin/env python
# coding: utf-8

# In[52]:


import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


# In[53]:


### Set up root directory
print (os.getcwd())
os.chdir('homepath')
print (os.getcwd())


# In[54]:


featurePath = "src/features/FeatureProcessing"


# In[55]:


coredf = pd.read_csv(os.path.join(featurePath,"core.csv"))
stringdf = pd.read_csv(os.path.join(featurePath,"string.csv"))
embeddingdf = pd.read_csv(os.path.join(featurePath,"embedding.csv"))
nlidf = pd.read_csv(os.path.join(featurePath,"nli.csv"))
wordmoverdf = pd.read_csv(os.path.join(featurePath,"wordmover.csv"))
domstringdf = pd.read_csv(os.path.join(featurePath,"domstring.csv"))
domEmbeddingdf = pd.read_csv(os.path.join(featurePath,"domEmbedding.csv"))


# In[56]:


coredf.head()


# In[57]:


df=pd.merge(coredf, stringdf, how='left',left_on=['id'],right_on=['id'])
df.columns


# In[58]:


df=pd.merge(df, embeddingdf, how='left',left_on=['id'],right_on=['id'])
df.columns


# In[59]:


df=pd.merge(df, nlidf, how='left',left_on=['id'],right_on=['id'])
df.columns


# In[60]:


df=pd.merge(df, domstringdf, how='left',left_on=['id'],right_on=['id'])
df.columns


# In[61]:


df=pd.merge(df, domEmbeddingdf, how='left',left_on=['id'],right_on=['id'])
df.columns


# In[62]:


df=pd.merge(df, wordmoverdf, how='left',left_on=['id'],right_on=['id'])
df.columns


# In[63]:


cuidf = pd.read_csv(os.path.join(featurePath,"cuidf.csv"))
cuidf.head()


# In[64]:


df=pd.merge(df, cuidf, how='left',left_on=['s1_join','s2_join','scores'],right_on=['a_join','b_join','score'])
df.columns


# In[65]:


print (len(df), len(df['cuisim'])- df['cuisim'].count())


# In[66]:


df.drop(['a_y','b_y','score','a_join','b_join'],axis=1,inplace=True)
df.rename(columns={'a_x':'a','b_x':'b'},inplace=True)


# In[67]:


df.to_csv(os.path.join(featurePath,"train.csv"),sep=",",index=False)


# In[ ]:





# In[ ]:




