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


# In[7]:


# Xfeatures = data[['CosineSentSim', 'corrSentSim','euSentSim', 'sqeuSentSim','cityblockSentSim','Jaccard','Jaccard_G','Q2', 'Q3', 'Q4', 'Cosine', 'Dice', 'Overlap',
#        'Tversky', 'MongeElkan','TfIdf','Affine', 'Bag','wordmover',
#        'Editex', 'Jaro', 'Levenshtein','NeedlemanWunsch','SmithWaterman',
#         'ModifiedESIM_2Class_Similar','ModifiedESIM_p2h_h2p_2Class_Similar','OriginalESIM_2Class_Similar','OriginalESIM_p2h_h2p_2Class_Similar',
#         'ModifiedESIM_3Class_Similar','ModifiedESIM_p2h_h2p_3Class_Similar','OriginalESIM_3Class_Similar','OriginalESIM_p2h_h2p_3Class_Similar']]
# Yfeatures = data[['scores']]


# In[8]:


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


# In[9]:


data[data['isGen']==0].head()


# In[10]:


Xfeatures = data[data['isGen']==0][features]
Yfeatures = data[data['isGen']==0][['scores']]


# In[11]:


# import seaborn as sns
# # visualize the relationship between the features and the response using scatterplots
# sns.pairplot(data, x_vars=['BioSentSim','Jaccard','Jaccard_G'], y_vars='scores', size=7, aspect=0.7)


# In[12]:


# sns.pairplot(data, x_vars=['BioSentSim','Jaccard','Jaccard_G'], y_vars='scores', size=7, aspect=0.7, kind='reg')


# In[13]:


Xfeatures.head()
Yfeatures.head()


# In[14]:


X = Xfeatures.values
Y = Yfeatures.values


# ### Data partitioning

# In[15]:


trainSize = 0.7
valSize = 0.2
testSize = 0.1
dataSize = X.shape[0]
print (dataSize)


# In[16]:


from sklearn.model_selection import train_test_split
XtrainData, XtestData, YtrainData, YtestData = train_test_split(X, Y, test_size=testSize, shuffle = False)
#XtrainData, XvalData, YtrainData, YvalData = train_test_split(XtrainData, YtrainData, test_size=valSize, shuffle = True)


# In[17]:


print (len(XtrainData), len(XtestData))
print (len(YtrainData), len(YtestData))
YtrainData = YtrainData.reshape(-1)
YtestData = YtestData.reshape(-1)


# In[18]:


print (XtrainData.shape)


# ### Handle for Generated Data

# In[19]:


if isGenerated:
    Xfeatures = data[data['isGen']==1][features].values
    Yfeatures = data[data['isGen']==1][['scores']].values.reshape(-1)
    
    print (Xfeatures.shape, Yfeatures.shape)
    XtrainData = np.concatenate((XtrainData,Xfeatures), axis=0)
    YtrainData = np.concatenate((YtrainData,Yfeatures), axis=0)


# In[20]:


print (XtrainData.shape,YtrainData.shape, XtestData.shape, YtestData.shape)


# ### Scaling

# In[21]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
XtrainDataScaled = scaler.fit_transform(XtrainData)
#XvalDataScaled = scaler.transform(XvalData)
XtestDataScaled = scaler.transform(XtestData)


# In[22]:


XtrainDataScaled.shape


# In[23]:


# #Feature Selection
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# XtrainDataScaled = SelectKBest(chi2, k=2).fit_transform(XtrainDataScaled, YtrainData)


# ### Training and GridSearch

# In[24]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
#from xgboost import XGBRegressor
import lightgbm as lgb


# In[25]:



# Create the parameter grid based on the results of random search 
# param_grid = {
#     'max_depth': [5, 10, 25, 50, 70, 80, 90, 100, 110],
#     'loss':['ls','lad','huber','quantile'],
#     #'max_features': [2, 3],
#     #'min_samples_leaf': [3, 4, 5],
#     #'min_samples_split': [8, 10, 12],
#     'n_estimators': [50, 100, 150, 200, 300, 500, 1000],
#     'learning_rate': [0.1,0.01,0.001,0.0001]
# }
# param_grid = {'min_child_weight':[4,5], 'gamma':[i/10.0 for i in range(3,6)],  'subsample':[i/10.0 for i in range(6,11)],
# 'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [2,3,4,5]}
param_grid = {#'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              #'learning_rate': [0.1, 0.01,.03, 0.05, .07,0.001], #so called `eta` value
              #'max_depth': [5, 6, 7,10,15],
              #'min_child_weight': [4],
              #'silent': [1],
              #'subsample': [0.7],
              #'colsample_bytree': [0.7],
              #'n_estimators':[50, 100, 150, 200, 300, 500, 1000],
            }


# In[26]:


lgb_train = lgb.Dataset(XtrainDataScaled, YtrainData)

params = {'task':'train', 'metric':{'auc'}}
#gbm = lgb.train(params, lgb_train)


# In[27]:


# Create a based model
model = lgb.LGBMRegressor(task='train')


# In[32]:


gridParams = {
    'learning_rate': [0.005, 0.01],
    'n_estimators': [8,16,24],
    'num_leaves': [6,8,12,16], # large num_leaves helps improve accuracy but might lead to over-fitting
    'boosting_type' : ['gbdt', 'dart'], # for better accuracy -> try dart
    'objective' : ['regression'],
    'max_bin':[255, 510], # large max_bin helps improve accuracy but might slow down training progress
    'random_state' : [500],
    'colsample_bytree' : [0.64, 0.65, 0.66],
    'subsample' : [0.7,0.75],
    'reg_alpha' : [1,1.2],
    'reg_lambda' : [1,1.2,1.4],
    }
'''
grid = GridSearchCV(mdl, gridParams, verbose=1, cv=4, n_jobs=-1)
# Run the grid
grid.fit(XtrainDataScaled, YtrainData)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)
'''


# In[34]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = gridParams, 
                          cv = 5, n_jobs = -1, verbose = 3)


# In[35]:


# Fit the grid search to the data
grid_search.fit(XtrainDataScaled, YtrainData)
grid_search.best_params_


# In[38]:


model = grid_search.best_estimator_


# In[39]:


print(model.feature_importances_)
for f,i in zip(features,model.feature_importances_):
    print(f,":",i)


# In[41]:


model.score(XtrainDataScaled, YtrainData)


# In[ ]:


# model.coef_


# In[42]:


#yValPred = model.predict(XvalDataScaled)
yTestPred = model.predict(XtestDataScaled)
yTrainPred = model.predict(XtrainDataScaled)


# In[43]:


#Clipping
#yValPred = np.clip(yValPred,0,5)
yTestPred = np.clip(yTestPred,0,5)
yTrainPred = np.clip(yTrainPred,0,5)


# In[44]:


print (mean_squared_error(YtrainData, yTrainPred))
#print (mean_squared_error(YvalData, yValPred))
print (mean_squared_error(YtestData, yTestPred))


# ### Output Results

# In[45]:


get_ipython().system('mkdir /tmp/lgbm')


# In[46]:


outpath = "/tmp/lgbm"
testg = open(os.path.join(outpath,"testgs.txt"),'w')
testp = open(os.path.join(outpath,"testsys.txt"),'w')
# valg = open(os.path.join(outpath,"valgs.txt"),'w')
# valp = open(os.path.join(outpath,"valsys.txt"),'w')
traing = open(os.path.join(outpath,"traings.txt"),'w')
trainp = open(os.path.join(outpath,"trainsys.txt"),'w')


# In[47]:


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


# In[48]:


testg.close()
testp.close()
# valg.close()
# valp.close()
traing.close()
trainp.close()


# In[60]:


get_ipython().system(' perl ~/nb/ASU-n2c2-2019/data/train/raw/correlation-noconfidence.pl /tmp/lgbm/testgs.txt /tmp/lgbm/testsys.txt')
get_ipython().system(' perl ~/nb/ASU-n2c2-2019/data/train/raw/correlation-noconfidence.pl /tmp/lgbm/traings.txt /tmp/lgbm/trainsys.txt')


# In[49]:


pd.DataFrame({'GoldScore':YtrainData.reshape(-1,),
              'PredScore':yTrainPred.reshape(-1,),
              'score':data.scores.head(yTrainPred.shape[0]),
              'a':data.a.head(yTrainPred.shape[0]),
              'b':data.b.head(yTrainPred.shape[0])},columns=['GoldScore','PredScore','score','a','b']).head()


# In[50]:


# predscore = np.concatenate((yTrainPred.reshape(-1,),yValPred.reshape(-1,),yTestPred.reshape(-1,)),axis=0)
# goldscore = np.concatenate((YtrainData.reshape(-1,),YvalData.reshape(-1,),YtestData.reshape(-1,)),axis=0)
predscore = np.concatenate((yTrainPred.reshape(-1,),yTestPred.reshape(-1,)),axis=0)
goldscore = np.concatenate((YtrainData.reshape(-1,),YtestData.reshape(-1,)),axis=0)
dt = data
dt['predScore'] = predscore
dt['goldScore'] = goldscore


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


dt.to_csv(open(os.path.join(outpath,"xgb.csv"),'w'),sep=',',index=False)


# In[ ]:





# In[ ]:




