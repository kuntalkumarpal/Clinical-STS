#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import pandas as pd
import numpy as np
import time
import os


# In[2]:


print (os.getcwd())
print (os.getcwd())
embeddingFile = 'BioSentEmbeddings.pkl'
#pathFeatures = 'models/token+sequence'
pathFeatures = 'src/data'
#fileFeatures = os.path.join(pathFeatures,'df.csv')
fileFeatures = os.path.join(pathFeatures,'TokenizedNewData.csv')


# In[3]:


data = pd.read_csv(fileFeatures,delimiter=',')


# In[4]:


data.head()


# In[5]:


data.columns


# In[6]:


#data.rename( columns={'Unnamed: 0':'id'}, inplace=True )


# In[7]:


data.head()


# In[35]:


Xfeatures = data[['CosineSentSim', 'corrSentSim','euSentSim', 'sqeuSentSim','wmdw2v','Jaccard','Jaccard_G','Q2', 'Q3', 'Q4', 'Cosine', 'Dice', 'Overlap',
       'Tversky', 'MongeElkan','TfIdf','Affine','Bag','Jaro','Editex','Levenshtein','NeedlemanWunsch','SmithWaterman']]
Yfeatures = data[['scores']]


# In[36]:


import seaborn as sns
# visualize the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['BioSentSim','Jaccard','Jaccard_G'], y_vars='scores', size=7, aspect=0.7)


# In[37]:


sns.pairplot(data, x_vars=['BioSentSim','Jaccard','Jaccard_G'], y_vars='scores', size=7, aspect=0.7, kind='reg')


# In[38]:


Xfeatures.head()
Yfeatures.head()


# In[39]:


X = Xfeatures.values
Y = Yfeatures.values


# ### Data partitioning

# In[40]:


trainSize = 0.7
valSize = 0.2
testSize = 0.1
dataSize = X.shape[0]
print (dataSize)


# In[41]:


from sklearn.model_selection import train_test_split
XtrainData, XtestData, YtrainData, YtestData = train_test_split(X, Y, test_size=testSize, shuffle = False)
XtrainData, XvalData, YtrainData, YvalData = train_test_split(XtrainData, YtrainData, test_size=valSize, shuffle = False)


# In[42]:


print (len(XtrainData), len(XvalData), len(XtestData))
print (len(YtrainData), len(YvalData), len(YtestData))


# In[43]:


print (XtrainData.shape)


# ### Scaling

# In[44]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
XtrainDataScaled = scaler.fit_transform(XtrainData)
XvalDataScaled = scaler.transform(XvalData)
XtestDataScaled = scaler.transform(XtestData)


# In[45]:


XtrainDataScaled.shape


# In[46]:


# #Feature Selection
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# XtrainDataScaled = SelectKBest(chi2, k=2).fit_transform(XtrainDataScaled, YtrainData)


# In[47]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[48]:


model = LinearRegression()
model = model.fit(XtrainDataScaled, YtrainData)


# In[49]:


model.score(XtrainDataScaled, YtrainData)


# In[50]:


model.coef_


# In[51]:


yValPred = model.predict(XvalDataScaled)
yTestPred = model.predict(XtestDataScaled)
yTrainPred = model.predict(XtrainDataScaled)


# In[52]:


#Clipping
yValPred = np.clip(yValPred,0,5)
yTestPred = np.clip(yTestPred,0,5)
yTrainPred = np.clip(yTrainPred,0,5)


# In[53]:


print (mean_squared_error(YtrainData, yTrainPred))
print (mean_squared_error(YvalData, yValPred))
print (mean_squared_error(YtestData, yTestPred))


# ### Output Results

# In[54]:


outpath = "output/LinearRegression"
testg = open(os.path.join(outpath,"testgs.txt"),'w')
testp = open(os.path.join(outpath,"testsys.txt"),'w')
valg = open(os.path.join(outpath,"valgs.txt"),'w')
valp = open(os.path.join(outpath,"valsys.txt"),'w')
traing = open(os.path.join(outpath,"traings.txt"),'w')
trainp = open(os.path.join(outpath,"trainsys.txt"),'w')


# In[55]:


for eachg, eachp in zip(yTestPred.flatten().tolist(),YtestData.flatten().tolist()):
    testg.write(str(eachp)+"\n")
    testp.write(str(eachg)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))

for eachg, eachp in zip(yValPred.flatten().tolist(),YvalData.flatten().tolist()):
    valg.write(str(eachp)+"\n")
    valp.write(str(eachg)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))

for eachg, eachp in zip(yTrainPred.flatten().tolist(),YtrainData.flatten().tolist()):
    traing.write(str(eachp)+"\n")
    trainp.write(str(eachg)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))
    


# In[56]:


testg.close()
testp.close()
valg.close()
valp.close()
traing.close()
trainp.close()


# In[57]:


pd.DataFrame({'GoldScore':YtrainData.reshape(-1,),
              'PredScore':yTrainPred.reshape(-1,),
              'score':data.scores.head(yTrainPred.shape[0]),
              'a':data.a.head(yTrainPred.shape[0]),
              'b':data.b.head(yTrainPred.shape[0])},columns=['GoldScore','PredScore','score','a','b']).head()


# In[58]:


predscore = np.concatenate((yTrainPred.reshape(-1,),yValPred.reshape(-1,),yTestPred.reshape(-1,)),axis=0)
goldscore = np.concatenate((YtrainData.reshape(-1,),YvalData.reshape(-1,),YtestData.reshape(-1,)),axis=0)
dt = data
dt['predScore'] = predscore
dt['goldScore'] = goldscore


# In[59]:


len(predscore)


# In[60]:


dt.head()


# In[61]:


dt.to_csv(open(os.path.join(outpath,"LinRegOutput.csv"),'w'),sep=',')


# In[ ]:




