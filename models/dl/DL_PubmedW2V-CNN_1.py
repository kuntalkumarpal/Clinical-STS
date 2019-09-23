#!/usr/bin/env python
# coding: utf-8

# ## A Dense Neural network with Word2Vec embedding trained on pubmed abstracts

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
import torch.nn.functional as F


# In[2]:


# Reproducibility
manualSeed = 42

np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are suing GPU
print("Torch is available:",torch.cuda.is_available())
if torch.cuda.is_available():
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)


    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# In[3]:


print (os.getcwd())
os.chdir('<homepath>')
print (os.getcwd())


# In[4]:


embedding = "data/embeddings/pubmed_s100w10_min.bin"
data = "<srcdata>"


# In[5]:



# #Load pretrained model (since intermediate data is not included, the model cannot be refined with additional data)
# #model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, norm_only=True)


# import gensim
# model = word2vec.KeyedVectors.load_word2vec_format(embedding, binary=True)

# print (model['man'].shape)
# # Deal with an out of dictionary word: Михаил (Michail)
# if 'cholangiocarcinoma' in model:
#     print(model['cholangiocarcinoma'].shape)
# else:
#     print('{0} is an out of dictionary word'.format('cholangiocarcinoma'))
    
# print(model.most_similar(positive=['woman', 'king'], negative=['man']))

# print(model.doesnt_match("breakfast cereal dinner lunch".split()))

# print(model.similarity('woman', 'man'))

# print(model.most_similar('cholangiocarcinoma'))
# print(model.most_similar('Tamoxifen'))


# ### DataLoader Class

# In[6]:


import src.data.DataLoader as CustomDataLoader
from src.features.CustomTokenizer import CustomTokenizer


# In[7]:


maxSentenceLen = 15
embedDim = 100
validation_split = .1
test_split = .2


# ### MESSY LOADER : TODO CLEANUPS

# In[8]:


class ClinicalSTS(Dataset):
    ''' Class to Load STS data '''
    
    def __init__(self, typeData):
        
        zeroTensor = torch.zeros(100)
        self.dataSize = 1654
        
        ''' Load Pretrained Word2Vec Model '''
        preTrainedWord2Vec = word2vec.KeyedVectors.load_word2vec_format(embedding, binary=True)

        ''' Load the data from the file '''
        #data = pd.read_csv(file, delimiter="\t", header=None,names=['a','b','score'])
        #fullData = np.loadtxt(data, delimiter='\t')
        pairs = CustomDataLoader.DataLoader()
        # Pairs will have the whole data in list of tupples {a, b, score}
        print (len(pairs))
        print (pairs[3][0], pairs[3][1],pairs[3][2])
        print(CustomTokenizer(pairs[3][0]))
        print(CustomTokenizer(pairs[3][1]))
        print(pairs[3][2])

        #TODO: Provide full data
        textData = pairs[:self.dataSize]
        
        #self.len = self.dataSize
        
        ''' Get the Tokenization and embeddings '''
        aData = []
        bData = []
        labels = []
        for eachData in tqdm(textData):
            a = CustomTokenizer(eachData[0])
            b = CustomTokenizer(eachData[1])
            label = float(eachData[2])
            aEmbed = []
            bEmbed = []
            for eachToken in a:
                if eachToken in preTrainedWord2Vec:
                    aEmbed.append(preTrainedWord2Vec[eachToken])
                    
            if len(aEmbed) < maxSentenceLen:
                aEmbed += [zeroTensor] * (maxSentenceLen - len(aEmbed))
            elif len(aEmbed) > maxSentenceLen:
                aEmbed = aEmbed[:maxSentenceLen]
            
            for eachToken in b:
                if eachToken in preTrainedWord2Vec:
                    bEmbed.append(preTrainedWord2Vec[eachToken])
                    
            if len(bEmbed) < maxSentenceLen:
                bEmbed += [zeroTensor] * (maxSentenceLen - len(bEmbed))
            elif len(bEmbed) > maxSentenceLen:
                bEmbed = bEmbed[:maxSentenceLen]
                
            aData.append(aEmbed)
            bData.append(bEmbed)
            labels.append(label)
                   
                
        #Check shapes and sizes
        print (len(aData), len(bData), len(labels))
        print (len(aData[3]), len (bData[3]), len (aData[3][0]),type(aData[3][0]))
        
        ''' Convert to Tensors'''
        aDataTensor = torch.FloatTensor(aData)
        bDataTensor = torch.FloatTensor(bData)
        labels = torch.FloatTensor(labels)
        print (aDataTensor.shape, bDataTensor.shape, labels.shape)
        
        xData = torch.cat((aDataTensor,bDataTensor),1)
        print (xData.shape)
        
        splitVal = int(np.floor(validation_split * self.dataSize))
        splitTest = int(np.floor(test_split * self.dataSize))
        self.xVal,  self.yVal    = xData[:splitVal],                    labels[:splitVal]
        self.xTest, self.yTest   = xData[splitVal: splitVal+splitTest], labels[splitVal: splitVal+splitTest]
        self.xTrain, self.yTrain = xData[splitVal+splitTest:],          labels[splitVal+splitTest:]
        print (len(self.xTrain),len(self.xVal),len(self.xTest))
        print (splitVal, splitTest)
        
        if typeData == "train":
            self.xData = self.xTrain
            self.labels = self.yTrain
        elif typeData == "test":
            self.xData = self.xTest
            self.labels = self.yTest
        else:
            self.xData = self.xVal
            self.labels = self.yVal
        self.len = len(self.xData)
        print (len(self.xData),len(self.labels))

        
    def __getitem__(self, index):
        return self.xData[index], self.labels[index]
        
    def __len__(self):
        return self.len


# ### Define Train and Test Loader

# In[9]:


stime = time.time()
datasetTrain = ClinicalSTS('train')
print("Time:",time.time()- stime)

stime = time.time()
datasetTest = ClinicalSTS('test')
print("Time:",time.time()- stime)

stime = time.time()
datasetVal = ClinicalSTS('val')
print("Time:",time.time()- stime)


# In[10]:


batchSize = 32

trainLoader = torch.utils.data.DataLoader(datasetTrain, batch_size=batchSize,shuffle=False)
validationLoader = torch.utils.data.DataLoader(datasetVal, batch_size=batchSize,shuffle=False)
testLoader = torch.utils.data.DataLoader(datasetTest, batch_size=batchSize,shuffle=False)


# In[11]:


print (len(trainLoader), len(validationLoader), len(testLoader))


# In[12]:


print (len(trainLoader.dataset), len(validationLoader.dataset), len(testLoader.dataset))


# ### Model Class

# In[30]:


class CNN(nn.Module):
    ''' A Dense Neural Network '''
    
    def __init__(self):
        super().__init__()
        self.modelName = "CNN_cW2V"
        self.c1 = nn.Conv2d(30, 32, kernel_size = 5)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(200,1)
        
    def forward(self, x):
        """ Layer Stacking and Logic Core """
        inpSize = x.shape[0]
        x = x.view(inpSize, 1, )
        #print ("InputSize:",inpSize)
        l1out = F.relu(self.mp(self.c1(x)))
        l2out = l1out.view(inpSize, -1)
        l3out = self.fc(l2out)
        return yPred
        


# ### Define Models, Loss, Optimizer

# In[31]:


model = CNN()


# In[32]:


lossCriterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.001)


# ## Training

# In[33]:


def train(epoch):
    model.train()
    loss = 0
    for batchIdx, data in enumerate(trainLoader):
        xTrain, labels = data
        yPred = model(xTrain)
        labels = labels.view(-1,1)
        #print (xTrain.shape,yPred.shape, labels.shape)
        loss = lossCriterion(yPred, labels)
        if batchIdx %10 == 0:
            print ("Epoch : {}, Batch : {}, Progress:{:.2f}% Loss: {:.6f}".format(epoch,batchIdx,batchIdx * 100./len(trainLoader), loss.data))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# ## Testing 

# In[34]:


#outpath = os.path.join("output/",model.modelName)
outpath = os.path.join("output/","CNN_cW2V")


# In[35]:


def test(loader, model=model):
#     if isVal:
#         loader = validationLoader
#     else:
#         loader = testLoader
    model.eval()
    testloss = 0
    allPredValue = []
    for batchIdx, data in enumerate(loader):
        xVal, labels = data
        yPred = model(xVal)
        labels = labels.view(-1,1)
        testloss += lossCriterion(yPred, labels).data
        allPredValue.append((labels.data.squeeze(),yPred.data.squeeze()))
        
    testloss /= len(loader)
    
    # print ("Average Loss: ",loss.data)
    
    return allPredValue, testloss.data
    


# In[36]:


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

    for gs,ps in output:
        for eachGs, eachPs in zip(gs,ps):
            #print (eachGs.item(), eachPs.item() )
            g.write(str(round(eachGs.item(),2))+"\n")
            s.write(str(round(np.clip(eachPs.item(),0,5),2))+"\n")
            
    g.close()
    s.close()
    


# In[37]:


epochs = 100
isVal = True
prevLoss = 0
evalloss = 0

# 1 - valid
# 2 - test
# 3 - train


for eachEpoch in range(1,epochs):
    train(eachEpoch)
    output, evalloss = test(validationLoader)
    print (evalloss)
    if eachEpoch > 1 and evalloss > prevLoss:
        continue
    
    print ("Found Better model with loss :{:.4f}, PreviousLoss:{:.4f}".format(evalloss,prevLoss))
    prevLoss = evalloss
    #Save the best model
    bestmodelFile = os.path.join(outpath,'bestModel.pt')
    pickle.dump(model,open(bestmodelFile,'wb'))
    
    #Output results in file
    results('val', output, evalloss, eachEpoch)
   


# ### Run Test results with the best model

# In[21]:



trainedModelFile = os.path.join(outpath,'bestModel.pt')
isVal = False
bestModel = pickle.load(open(trainedModelFile,'rb'))
output, evalloss = test(testLoader,model)
results('test', output, evalloss, 0)
print("Test Results:\n Loss:{:.4f}".format(evalloss))

output, evalloss = test(trainLoader,model)
results('train', output, evalloss, 0)
print("Train Results:\n Loss:{:.4f}".format(evalloss))


# In[81]:


np.clip(4.3,0,5)


# In[ ]:




