#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import os
import csv
import numpy


# In[10]:


file = "../data/train/raw/filename"


# In[11]:


data = pd.read_csv(file, delimiter="\t", header=None,names=['a','b','score'])


# In[12]:


data.head()


# In[13]:


len(data)


# In[14]:


data.iloc[:,2].unique()


# In[15]:


a = data.iloc[:,2].unique()
a.sort()
print(a)


# In[16]:


len(data[data.score>=4.5])


# In[17]:


data[data.score>=4.5].head()


# In[18]:


len(data[data.score<1])


# In[19]:


data[data.score<1].head()


# In[20]:


# Data issue probably (Nothing in bracket)
data[data.a.str.contains("\[\]")]


# # Medical names (208 data col='a' or col='b' or both)
# ## Q: Which medDatabase to get the similarity between them ?

# In[21]:


data[data.a.str.contains("\[*\]")|data.b.str.contains("\[*\]")]


# # No marked explicit medical names

# In[22]:


medNameCols = data.a.str.contains("\[*\]")|data.b.str.contains("\[*\]")
data[~medNameCols]


# # Contains numeric values (probably dosage) #439
# ## Q : Need to check whether change in dosage makes a and b dissimilar or not

# In[23]:


data[data.a.str.contains("[0-9]")]


# In[24]:


medNameNumericCols = data.a.str.contains("\[*\]")|data.b.str.contains("\[*\]")|data.a.str.contains("[0-9]")|data.b.str.contains("[0-9]")
data[~medNameNumericCols]


# ## Statements related to time

# In[86]:


timeKeyWords = "hours|hour|day|days|mins|minutes|minute|daily|time"
data[data.a.str.contains(timeKeyWords)|data.b.str.contains(timeKeyWords)]


# ## Statements related to Dosage

# In[94]:


timeKeyWords = " mg| mcg|gram|tablet|tablets|dose|dosage|tab"
data[data.a.str.lower().str.contains(timeKeyWords)|data.b.str.lower().str.contains(timeKeyWords)]


# In[33]:


data[(data.a.str.contains("\[*\]")|data.b.str.contains("\[*\]"))&(data.score>3)]


# In[ ]:




