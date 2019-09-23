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
print (os.getcwd())


# In[3]:


df = pd.read_csv("src/features/FeatureProcessing/multiRegressionOP.csv")


# In[4]:


df.head()


# In[5]:


df.columns


# In[7]:


Xfeatures = df[['lr', 'ada', 'et', 'xgb', 'gb', 'rf']]
Yfeatures = df[['gold']]


# In[8]:


X = Xfeatures.values
Y = Yfeatures.values


# ### Data partitioning

# In[9]:


trainSize = 0.7
valSize = 0.2
testSize = 0.1
dataSize = X.shape[0]
print (dataSize)


# In[10]:


from sklearn.model_selection import train_test_split
XtrainData, XtestData, YtrainData, YtestData = train_test_split(X, Y, test_size=testSize, shuffle = False)
# XtrainData, XvalData, YtrainData, YvalData = train_test_split(XtrainData, YtrainData, test_size=valSize, shuffle = False)


# In[12]:


print (len(XtrainData), len(XtestData))
print (len(YtrainData), len(YtestData))


# In[13]:


print (XtrainData.shape)


# In[15]:


YtrainData = YtrainData.reshape(-1,)
print(YtrainData.shape)
YtestData = YtestData.reshape(-1)


# ### Training and GridSearch

# In[25]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error


# In[48]:



# Create the parameter grid based on the results of random search 
param_grid = {
#     'alpha_1' :[0.01,0.001,0.0001,0.00001], 
#     'alpha_2':[0.01,0.001,0.0001,0.00001], 
#     'lambda_1':[0.01,0.001,0.0001,0.00001],
#     'lambda_2':[0.01,0.001,0.0001,0.00001]
    'alpha_1' :[10], 
    'alpha_2':[10], 
}


# In[49]:



# Create a based model
rf = BayesianRidge()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)


# In[50]:


# Fit the grid search to the data
grid_search.fit(XtrainData, YtrainData)
grid_search.best_params_


# In[51]:


model = grid_search.best_estimator_


# In[52]:


# print(model.feature_importances_)
# for f,i in zip(features,model.feature_importances_):
#     print(f,":",i)


# In[53]:


model.score(XtrainData, YtrainData)


# In[54]:


model.coef_


# In[55]:


#yValPred = model.predict(XvalDataScaled)
yTestPred = model.predict(XtestData)
yTrainPred = model.predict(XtrainData)


# In[56]:


#Clipping
#yValPred = np.clip(yValPred,0,5)
yTestPred = np.clip(yTestPred,0,5)
yTrainPred = np.clip(yTrainPred,0,5)


# In[57]:


print (mean_squared_error(YtrainData, yTrainPred))
#print (mean_squared_error(YvalData, yValPred))
print (mean_squared_error(YtestData, yTestPred))


# In[58]:


outpath = "output/ensemble"
testg = open(os.path.join(outpath,"testgs.txt"),'w')
testp = open(os.path.join(outpath,"testsys.txt"),'w')
# valg = open(os.path.join(outpath,"valgs.txt"),'w')
# valp = open(os.path.join(outpath,"valsys.txt"),'w')
traing = open(os.path.join(outpath,"traings.txt"),'w')
trainp = open(os.path.join(outpath,"trainsys.txt"),'w')


# In[59]:


for eachg, eachp in zip(yTestPred.flatten().tolist(),YtestData.flatten().tolist()):
    testg.write(str(eachp)+"\n")
    testp.write(str(eachg)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))

# for eachg, eachp in zip(yValPred.flatten().tolist(),YvalData.flatten().tolist()):
#     valg.write(str(eachp)+"\n")
#     valp.write(str(eachg)+"\n")
#     #print (str(round(eachg,2)),str(round(eachp,2)))

for eachg, eachp in zip(yTrainPred.flatten().tolist(),YtrainData.flatten().tolist()):
    traing.write(str(eachp)+"\n")
    trainp.write(str(eachg)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))
    


# In[60]:


testg.close()
testp.close()
# valg.close()
# valp.close()
traing.close()
trainp.close()


# In[61]:


pd.DataFrame({'GoldScore':YtrainData.reshape(-1,),
              'PredScore':yTrainPred.reshape(-1,),
              'score':df.gold.head(yTrainPred.shape[0]),
              'a':df.a.head(yTrainPred.shape[0]),
              'b':df.b.head(yTrainPred.shape[0])},columns=['GoldScore','PredScore','score','a','b']).head()


# In[62]:


# predscore = np.concatenate((yTrainPred.reshape(-1,),yValPred.reshape(-1,),yTestPred.reshape(-1,)),axis=0)
# goldscore = np.concatenate((YtrainData.reshape(-1,),YvalData.reshape(-1,),YtestData.reshape(-1,)),axis=0)
predscore = np.concatenate((yTrainPred.reshape(-1,),yTestPred.reshape(-1,)),axis=0)
goldscore = np.concatenate((YtrainData.reshape(-1,),YtestData.reshape(-1,)),axis=0)
dt = df
dt['predScore'] = predscore
dt['goldScore'] = goldscore


# In[63]:


dt.to_csv(open(os.path.join(outpath,"ensemble.csv"),'w'),sep=',')


# ### Average as output

# In[64]:


outpath = "output/ensemble"
testg = open(os.path.join(outpath,"testgs.txt"),'w')
testp = open(os.path.join(outpath,"testsys.txt"),'w')
# valg = open(os.path.join(outpath,"valgs.txt"),'w')
# valp = open(os.path.join(outpath,"valsys.txt"),'w')
traing = open(os.path.join(outpath,"traings.txt"),'w')
trainp = open(os.path.join(outpath,"trainsys.txt"),'w')


# In[66]:


Xfeatures = df[['avgPred']]
Yfeatures = df[['gold']]


# In[67]:


X = Xfeatures.values
Y = Yfeatures.values


# In[69]:


from sklearn.model_selection import train_test_split
XtrainPred, XtestPred, YtrainGold, YtestGold = train_test_split(X, Y, test_size=testSize, shuffle = False)
# XtrainData, XvalData, YtrainData, YvalData = train_test_split(XtrainData, YtrainData, test_size=valSize, shuffle = False)


# In[71]:


for eachg, eachp in zip(XtestPred.flatten().tolist(),YtestGold.flatten().tolist()):
    testg.write(str(eachp)+"\n")
    testp.write(str(eachg)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))

# for eachg, eachp in zip(yValPred.flatten().tolist(),YvalData.flatten().tolist()):
#     valg.write(str(eachp)+"\n")
#     valp.write(str(eachg)+"\n")
#     #print (str(round(eachg,2)),str(round(eachp,2)))

for eachg, eachp in zip(XtrainPred.flatten().tolist(),YtrainGold.flatten().tolist()):
    traing.write(str(eachp)+"\n")
    trainp.write(str(eachg)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))
    


# In[72]:


testg.close()
testp.close()
# valg.close()
# valp.close()
traing.close()
trainp.close()


# In[ ]:




