#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import pandas as pd
import numpy as np
import time
import os
seed_value = 0
np.random.seed(seed_value)


# In[2]:


isGenerated = False


# In[3]:


print (os.getcwd())
print (os.getcwd())
embeddingFile = 'BioSentEmbeddings.pkl'
#pathFeatures = 'models/token+sequence'
pathFeatures = 'src/features/FeatureProcessing'
#fileFeatures = os.path.join(pathFeatures,'df.csv')
fileFeatures = os.path.join(pathFeatures,'train.csv')


# In[4]:


data = pd.read_csv(fileFeatures,delimiter=',')


# In[5]:


data.head()


# In[6]:


data.columns


# In[7]:


# a = ['CosineSentSim', 'euSentSim', 'sqeuSentSim', 'corrSentSim',
#        'cityblockSentSim', 'wmdw2v',
#        'Sequence1', 'Sequence2', 'Jaccard', 'Jaccard_G', 'Q2', 'Q3', 'Q4', 'Cosine',
#        'Dice', 'Overlap', 'Tversky', 'MongeElkan', 'TfIdf', 'Affine', 'Bag',
#        'Editex', 'Jaro', 'Levenshtein', 'NeedlemanWunsch', 'SmithWaterman',
#      'ModifiedESIM_2Class_Similar',
#        'ModifiedESIM_3Class_Similar', 
#        'ModifiedESIM_p2h_h2p_2Class_Similar',
#        'ModifiedESIM_p2h_h2p_3Class_Similar',
#        'OriginalESIM_2Class_Similar',  'OriginalESIM_3Class_Similar',
#        'OriginalESIM_p2h_h2p_2Class_Similar',
#        'OriginalESIM_p2h_h2p_3Class_Similar', 'Number_similarity']
# len(a)


# In[8]:


# Xfeatures = data[['CosineSentSim', 'corrSentSim','euSentSim', 'sqeuSentSim','cityblockSentSim','Jaccard','Jaccard_G','Q2', 'Q3', 'Q4', 'Cosine', 'Dice', 'Overlap',
#        'Tversky', 'MongeElkan','TfIdf','Affine', 'Bag','wordmover',
#        'Editex', 'Jaro', 'Levenshtein','NeedlemanWunsch','SmithWaterman',
#         'ModifiedESIM_2Class_Similar','ModifiedESIM_p2h_h2p_2Class_Similar','OriginalESIM_2Class_Similar','OriginalESIM_p2h_h2p_2Class_Similar',
#         'ModifiedESIM_3Class_Similar','ModifiedESIM_p2h_h2p_3Class_Similar','OriginalESIM_3Class_Similar','OriginalESIM_p2h_h2p_3Class_Similar']]
# Yfeatures = data[['scores']]


# In[9]:


features = ['Jaccard', 'Jaccard_G', 'Q2', 'Q3',
       'Q4', 'Cosine', 'Dice', 'Overlap', 'Tversky', 'MongeElkan', 'TfIdf',
       'Affine', 'Bag', 'Editex', 'Jaro', 'Levenshtein', 
            'NeedlemanWunsch',
       'SmithWaterman', 'CosineSentSim', 'euSentSim', 'sqeuSentSim',
       'corrSentSim', 
#             'cityblockSentSim', 
#         'ModifiedESIM_2Class_Similar','ModifiedESIM_p2h_h2p_2Class_Similar','OriginalESIM_2Class_Similar','OriginalESIM_p2h_h2p_2Class_Similar',
#         'ModifiedESIM_3Class_Similar','ModifiedESIM_p2h_h2p_3Class_Similar','OriginalESIM_3Class_Similar','OriginalESIM_p2h_h2p_3Class_Similar',
                 'DomJaccard', 'DomJaccard_G',
       'DomQ2', 'DomQ3', 'DomQ4', 'DomCosine', 'DomDice', 'DomOverlap',
       'DomTversky', 'DomMongeElkan', 'DomTfIdf',  'DomAffine', 'DomBag', 'DomEditex', 'DomJaro',
       'DomLevenshtein', 'DomNeedlemanWunsch', 'DomSmithWaterman',
       'DomWordmover',  'DomEmbCosine',
       'DomEmbEuclidean', 'DomEmbSqEuclidean', 'DomEmbCorrelation',
       'DomEmbCityblock',  
#             'wordmover',
            'cuisim'
           ]


# In[10]:


data[data['isGen']==0].head()


# In[11]:


Xfeatures = data[data['isGen']==0][features]
Yfeatures = data[data['isGen']==0][['scores']]


# In[12]:


# import seaborn as sns
# # visualize the relationship between the features and the response using scatterplots
# sns.pairplot(data, x_vars=['BioSentSim','Jaccard','Jaccard_G'], y_vars='scores', size=7, aspect=0.7)


# In[13]:


# sns.pairplot(data, x_vars=['BioSentSim','Jaccard','Jaccard_G'], y_vars='scores', size=7, aspect=0.7, kind='reg')


# In[14]:


Xfeatures.head()
Yfeatures.head()


# In[15]:


X = Xfeatures.values
Y = Yfeatures.values


# ### Data partitioning

# In[16]:


# trainSize = 0.7
# valSize = 0.2
# testSize = 0.1
# dataSize = X.shape[0]
# print (dataSize)


# In[17]:


# from sklearn.model_selection import train_test_split
# XtrainData, XtestData, YtrainData, YtestData = train_test_split(X, Y, test_size=testSize, shuffle = False)
# #XtrainData, XvalData, YtrainData, YvalData = train_test_split(XtrainData, YtrainData, test_size=valSize, shuffle = True)


# In[18]:


# print (len(XtrainData), len(XtestData))
# print (len(YtrainData), len(YtestData))
# YtrainData = YtrainData.reshape(-1)
# YtestData = YtestData.reshape(-1)


# In[19]:


# print (XtrainData.shape)


# In[20]:


XtrainData = X
YtrainData = Y.reshape(-1)


# In[21]:


print (XtrainData.shape, YtrainData.shape)


# ### Handle for Generated Data

# In[22]:


# if isGenerated:
#     Xfeatures = data[data['isGen']==1][features].values
#     Yfeatures = data[data['isGen']==1][['scores']].values.reshape(-1)
    
#     print (Xfeatures.shape, Yfeatures.shape)
#     XtrainData = np.concatenate((XtrainData,Xfeatures), axis=0)
#     YtrainData = np.concatenate((YtrainData,Yfeatures), axis=0)


# In[23]:


# print (XtrainData.shape,YtrainData.shape, XtestData.shape, YtestData.shape)


# ### Test Data Load

# In[24]:


tstdata = pd.read_csv('src/features/FeatureProcessing/test.csv',delimiter=',')


# In[25]:


tstdata.shape


# In[26]:


XtestData = tstdata[features].values


# In[27]:


XtestData.shape


# ### Scaling

# In[28]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
XtrainDataScaled = scaler.fit_transform(XtrainData)
#XvalDataScaled = scaler.transform(XvalData)
XtestDataScaled = scaler.transform(XtestData)


# In[29]:


XtrainDataScaled.shape


# In[30]:


# #Feature Selection
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# XtrainDataScaled = SelectKBest(chi2, k=2).fit_transform(XtrainDataScaled, YtrainData)


# ### Training and GridSearch

# In[31]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor


# In[32]:



# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [5, 10, 25, 50, 70, 80, 90, 100, 110],
    #'max_features': [2, 3],
    #'min_samples_leaf': [3, 4, 5],
    #'min_samples_split': [8, 10, 12],
    'n_estimators': [50, 100, 150, 200, 300, 500, 1000]
}


# In[33]:



# Create a based model
rf = ExtraTreesRegressor(random_state=seed_value)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)


# In[34]:


# Fit the grid search to the data
grid_search.fit(XtrainDataScaled, YtrainData)
grid_search.best_params_


# In[35]:


model = grid_search.best_estimator_


# In[37]:


print(model.feature_importances_)
for f,i in zip(features,model.feature_importances_):
    print(f,":",i)


# In[38]:


model.score(XtrainDataScaled, YtrainData)


# In[39]:


# model.coef_


# In[40]:


#yValPred = model.predict(XvalDataScaled)
yTestPred = model.predict(XtestDataScaled)
yTrainPred = model.predict(XtrainDataScaled)


# In[41]:


#Clipping
#yValPred = np.clip(yValPred,0,5)
yTestPred = np.clip(yTestPred,0,5)
yTrainPred = np.clip(yTrainPred,0,5)


# In[42]:


print (mean_squared_error(YtrainData, yTrainPred))
#print (mean_squared_error(YvalData, yValPred))
#print (mean_squared_error(YtestData, yTestPred))


# In[43]:


# yTestPred


# ### Output Results

# In[44]:


outpath = "output/et/test1"
testg = open(os.path.join(outpath,"testgs.txt"),'w')
testp = open(os.path.join(outpath,"testsys.txt"),'w')
# valg = open(os.path.join(outpath,"valgs.txt"),'w')
# valp = open(os.path.join(outpath,"valsys.txt"),'w')
traing = open(os.path.join(outpath,"traings.txt"),'w')
trainp = open(os.path.join(outpath,"trainsys.txt"),'w')


# In[45]:


# yTestPred.flatten().tolist()


# In[46]:


for  eachp in yTestPred.flatten().tolist():
    #testg.write(str(eachp)+"\n")
    testp.write(str(eachp)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))

# for eachg, eachp in zip(yValPred.flatten().tolist(),YvalData.flatten().tolist()):
#     valg.write(str(eachp)+"\n")
#     valp.write(str(eachg)+"\n")
#     #print (str(round(eachg,2)),str(round(eachp,2)))

for eachg, eachp in zip(yTrainPred.flatten().tolist(),YtrainData.flatten().tolist()):
    traing.write(str(eachp)+"\n")
    trainp.write(str(eachg)+"\n")
    #print (str(round(eachg,2)),str(round(eachp,2)))
    


# In[47]:


testg.close()
testp.close()
# valg.close()
# valp.close()
traing.close()
trainp.close()


# In[49]:


# pd.DataFrame({'GoldScore':YtrainData.reshape(-1,),
#               'PredScore':yTrainPred.reshape(-1,),
#               'score':data.scores.head(yTrainPred.shape[0]),
#               'a':data.a.head(yTrainPred.shape[0]),
#               'b':data.b.head(yTrainPred.shape[0])},columns=['GoldScore','PredScore','score','a','b']).head()


# In[50]:


# predscore = np.concatenate((yTrainPred.reshape(-1,),yValPred.reshape(-1,),yTestPred.reshape(-1,)),axis=0)
# goldscore = np.concatenate((YtrainData.reshape(-1,),YvalData.reshape(-1,),YtestData.reshape(-1,)),axis=0)
# predscore = np.concatenate((yTrainPred.reshape(-1,),yTestPred.reshape(-1,)),axis=0)
# goldscore = np.concatenate((YtrainData.reshape(-1,),YtestData.reshape(-1,)),axis=0)
predscore = yTestPred.reshape(-1,)
dt = tstdata
dt['predScore'] = predscore
#dt['goldScore'] = goldscore


# In[51]:


len(predscore)


# In[52]:


dt.tail(20)


# In[53]:


dt['predClass'] = dt.apply(lambda x: round(x['predScore']),axis=1)
dt.tail()


# In[54]:


dt['isClassMatch']=dt.apply(lambda x: x['predClass']==x['classes'],axis=1)


# In[55]:


dt.head()


# In[56]:


len(dt[dt.isClassMatch])*100.0 / len(dt)


# In[57]:


dt.to_csv(open(os.path.join(outpath,"et.csv"),'w'),sep=',',index=False)


# In[ ]:




