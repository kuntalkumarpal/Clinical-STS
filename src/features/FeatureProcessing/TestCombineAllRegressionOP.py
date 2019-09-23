#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import json
import numpy as np


# In[2]:


### Set up root directory
print (os.getcwd())
os.chdir('homepath')
print (os.getcwd())


# In[3]:


dflr = pd.read_csv("output/lr/test1/lr.csv")
dflr=dflr[['id', 'a', 'b', 'scores','predScore']]
dflr.head()


# In[4]:


dfada = pd.read_csv("output/ada/test1/ada.csv")
dfada=dfada[['id', 'a', 'b', 'scores','predScore']]
dfada.head()


# In[5]:


dfet = pd.read_csv("output/et/test1/et.csv")
dfet=dfet[['id', 'a', 'b', 'scores','predScore']]
dfet.head()


# In[6]:


dfxgb = pd.read_csv("output/xgb/test1/xgb.csv")
dfxgb=dfxgb[['id', 'a', 'b', 'scores','predScore']]
dfxgb.head()


# In[7]:


dfgb = pd.read_csv("output/gb/test/gb.csv")
dfgb=dfgb[['id', 'a', 'b', 'scores','predScore']]
dfgb.head()


# In[8]:


dfrf = pd.read_csv("output/RandomForestRegression/test1/rf.csv")
dfrf=dfrf[['id', 'a', 'b', 'scores','predScore']]
dfrf.head()


# In[9]:


dflasso = pd.read_csv("output/lasso/test1/lasso.csv")
dflasso=dflasso[['id', 'a', 'b', 'scores','predScore']]
dflasso.head()


# In[10]:


dfbr = pd.read_csv("output/BayesianRidgeRegression/test1/bayesianridge.csv")
dfbr=dfbr[['id', 'a', 'b', 'scores','predScore']]
dfbr.head()


# ### Combine

# In[11]:


df = pd.merge(dflr,dfada, how='left',left_on=['id'],right_on=['id'])
df.drop(['a_y','b_y','scores_y'],axis=1,inplace=True)
df.rename(columns={'a_x':'a','b_x':'b','scores_x':'gold','predScore_x':'lr','predScore_y':'ada'},inplace=True)


# In[12]:


df=pd.merge(df,dfet, how='left',left_on=['id'],right_on=['id'])
df.drop(['a_y','b_y','scores'],axis=1,inplace=True)
df.rename(columns={'a_x':'a','b_x':'b','predScore':'et'},inplace=True)


# In[13]:


df=pd.merge(df,dfxgb, how='left',left_on=['id'],right_on=['id'])
df.drop(['a_y','b_y','scores'],axis=1,inplace=True)
df.rename(columns={'a_x':'a','b_x':'b','predScore':'xgb'},inplace=True)


# In[14]:


df=pd.merge(df,dfgb, how='left',left_on=['id'],right_on=['id'])
df.drop(['a_y','b_y','scores'],axis=1,inplace=True)
df.rename(columns={'a_x':'a','b_x':'b','predScore':'gb'},inplace=True)


# In[15]:


df=pd.merge(df,dfrf, how='left',left_on=['id'],right_on=['id'])
df.drop(['a_y','b_y','scores'],axis=1,inplace=True)
df.rename(columns={'a_x':'a','b_x':'b','predScore':'rf'},inplace=True)


# In[16]:


df=pd.merge(df,dflasso, how='left',left_on=['id'],right_on=['id'])
df.drop(['a_y','b_y','scores'],axis=1,inplace=True)
df.rename(columns={'a_x':'a','b_x':'b','predScore':'lasso'},inplace=True)


# In[17]:


df=pd.merge(df,dfbr, how='left',left_on=['id'],right_on=['id'])
df.drop(['a_y','b_y','scores'],axis=1,inplace=True)
df.rename(columns={'a_x':'a','b_x':'b','predScore':'br'},inplace=True)


# In[18]:


df.head(20)


# In[19]:


df['avgPred'] = (df['lr']+df['ada']+df['xgb']+df['gb']+df['et']+df['rf']+df['lasso']+df['br'])/8.0
df['minPred'] = df[['lr','ada','xgb','gb','et','rf','lasso','br']].min(axis=1)
df['maxPred'] = df[['lr','ada','xgb','gb','et','rf','lasso','br']].max(axis=1)


# In[20]:


df.head(20)


# In[21]:


df.tail(20)


# In[22]:


df.to_csv("src/features/FeatureProcessing/multiRegressionOPTest1.csv",index=False)


# In[ ]:




