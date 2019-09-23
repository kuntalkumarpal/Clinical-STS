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
os.chdir('homepath')
print (os.getcwd())
pathFeatures = 'models/token+sequence'
fileFeatures = os.path.join(pathFeatures,'df.csv')


# In[3]:


data = pd.read_csv(fileFeatures,delimiter=',')


# In[4]:


data.head()


# In[5]:


data.columns


# In[6]:


data.rename( columns={'Unnamed: 0':'id'}, inplace=True )


# In[7]:


data.head()


# In[8]:


Xfeatures = data[['Jaccard','Jaccard_G','Q2', 'Q3', 'Q4', 'Cosine', 'Dice', 'Overlap',
       'Tversky', 'MongeElkan','TfIdf','Affine','Bag','Jaro','Editex','Levenshtein','NeedlemanWunsch','SmithWaterman']]
Yfeatures = data[['Score']]


# In[9]:


Xfeatures.head()
Yfeatures.head()


# In[10]:


scaledData = Xfeatures.values
Y = Yfeatures.values


# ### Scaling

# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(scaledData)


# In[12]:


trainSize = 0.7
valSize = 0.2
testSize = 0.1
dataSize = X.shape[0]
print (dataSize)


# In[13]:


from sklearn.model_selection import train_test_split
XtrainData, XtestData, YtrainData, YtestData = train_test_split(X, Y, test_size=testSize)
XtrainData, XvalData, YtrainData, YvalData = train_test_split(XtrainData, YtrainData, test_size=valSize)


# In[14]:


print (len(XtrainData), len(XvalData), len(XtestData))
print (len(YtrainData), len(YvalData), len(YtestData))


# In[17]:


XtrainData.shape


# In[23]:


numFeatures = XtrainData.shape[1]


# In[19]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import gensim.models as word2vec
from tqdm import tqdm
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
import pickle


# In[55]:


XTrain = torch.from_numpy(XtrainData).type(torch.FloatTensor)
XTest = torch.from_numpy(XtestData).type(torch.FloatTensor)
XVal = torch.from_numpy(XvalData).type(torch.FloatTensor)
YTrain = torch.from_numpy(YtrainData).type(torch.FloatTensor)
YTest = torch.from_numpy(YtestData).type(torch.FloatTensor)
YVal = torch.from_numpy(YvalData).type(torch.FloatTensor)

XTrain.type()


# In[164]:


print (XTrain.shape, YTrain.shape, XVal.shape, YVal.shape, XTest.shape, YTest.shape)


# In[65]:


# batchSize = 128

# trainLoader = torch.utils.data.DataLoader(XTrain, batch_size=batchSize,shuffle=False)
# validationLoader = torch.utils.data.DataLoader(XVal, batch_size=batchSize,shuffle=False)
# testLoader = torch.utils.data.DataLoader(XTest, batch_size=batchSize,shuffle=False)


# ### Model Class

# In[57]:


class NN(nn.Module):
    ''' A Dense Neural Network '''
    
    def __init__(self):
        super().__init__()
        self.modelName = "OneLyrNeuralNet"
        self.l1 = nn.Linear(numFeatures,1)
        
        
    def forward(self, x):
        """ Layer Stacking and Logic Core """
        #print ("Before:",x.shape)
        x = x.view(-1, numFeatures)  
        #print (x.shape)
        yPred = self.l1(x)
        return yPred
        


# ### Define Models, Loss, Optimizer

# In[156]:


model = NN()


# In[157]:


lossCriterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.07)


# ## Training

# In[158]:


def train(epoch):
    model.train()
    loss = 0
#     for batchIdx, data in enumerate([XTrain,YTrain]):
#         print (len(data))
#         xTrain, labels = data[0], data[1]
#         yPred = model(xTrain)
#         labels = labels.view(-1,1)
#         #print (xTrain.shape,yPred.shape, labels.shape)
#         loss = lossCriterion(yPred, labels)
#         if batchIdx %10 == 0:
#             print ("Epoch : {}, Batch : {}, Progress:{:.2f}% Loss: {:.6f}".format(epoch,batchIdx,batchIdx * 100./len(trainLoader), loss.data))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    yPred = model(XTrain)
    loss = lossCriterion(yPred, YTrain)
    #print ("Epoch : {}, Loss: {:.6f}".format(epoch, loss.data))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ## Testing 

# In[159]:


#outpath = os.path.join("output/",model.modelName)
outpath = os.path.join("output","NeuralNetSingle")


# In[160]:


def test(loader, model=model):
    model.eval()
    testloss = 0
    allPredValue = []
#     for batchIdx, data in enumerate(loader):
#         xVal, labels = data
#         yPred = model(xVal)
#         labels = labels.view(-1,1)
#         testloss += lossCriterion(yPred, labels).data
#         allPredValue.append((labels.data.squeeze(),yPred.data.squeeze()))
        
#     testloss /= len(loader)
    
#     # print ("Average Loss: ",loss.data)
    x, y = loader[0], loader[1]
    yPred = model(x)
    testloss = lossCriterion(yPred, y).data
    allPredValue = (y.data.squeeze(),yPred.data.squeeze())
    return allPredValue, testloss
    


# In[161]:


# Write results
def results(typeData, output, loss, epoch):
    
        
    gsc = os.path.join(outpath,typeData+"gs.txt")
    sys = os.path.join(outpath,typeData+"sys.txt")
    results = os.path.join(outpath,typeData+"result.txt")
        
    rs = open(results,'w')
    rs.write("Epoch:{},\n Loss:{:.4f}\n".format(epoch,loss))
    rs.close()
    
    
    g = open(gsc,'w')
    s = open(sys,'w')

    for eachGs, eachPs in zip(output[0],output[1]):
        #print (eachGs.item(), eachPs.item() )
        g.write(str(round(eachGs.item(),2))+"\n")
        s.write(str(round(np.clip(eachPs.item(),0,5),2))+"\n")
            
    g.close()
    s.close()
    


# In[162]:


epochs = 500
isVal = True
prevLoss = 0
evalloss = 0

# 1 - valid
# 2 - test
# 3 - train


for eachEpoch in range(1,epochs):
    train(eachEpoch)
    output, evalloss = test((XVal,YVal))
    print (evalloss)
    if eachEpoch > 1 and evalloss > prevLoss:
        continue
    
    print ("Epoch:{} Found Better model with loss :{:.4f}, PreviousLoss:{:.4f}".format(eachEpoch,evalloss,prevLoss))
    prevLoss = evalloss
    #Save the best model
    bestmodelFile = os.path.join(outpath,'bestModel.pt')
    pickle.dump(model,open(bestmodelFile,'wb'))
    
    #print (output)
    #Output results in file
    results('val', output, evalloss, eachEpoch)
   


# ### Run Test results with the best model

# In[163]:



trainedModelFile = os.path.join(outpath,'bestModel.pt')
isVal = False
bestModel = pickle.load(open(trainedModelFile,'rb'))
output, evalloss = test((XTest, YTest),model)
results('test', output, evalloss, 0)
print("Test Results:\n Loss:{:.4f}".format(evalloss))

output, evalloss = test((XTrain, YTrain),model)
results('train', output, evalloss, 0)
print("Train Results:\n Loss:{:.4f}".format(evalloss))


# In[ ]:




