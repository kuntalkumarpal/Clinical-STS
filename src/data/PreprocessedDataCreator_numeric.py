#!/usr/bin/env python
# coding: utf-8

# ## Analysis and Validation of our Custom Tokenizer

# ### Handling Less overlap cases - Where overlap is one word and rest is different. OverlapCoeff will not handle this as length is small

# ## Data Creator with all Pre-processing and Scores Rounding Off

# In[1]:


import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


# In[2]:


### Set up root directory
print (os.getcwd())
os.chdir('/home/kkpal/ASU-n2c2-2019')
print (os.getcwd())


# In[3]:


import src.data.DataLoader as CustomDataLoader
from src.features.CustomTokenizer import CustomTokenizer


# In[4]:


pairs = CustomDataLoader.DataLoader()


# In[5]:


len(pairs)


# In[6]:


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return int(np.floor(n*multiplier + 0.5) / multiplier)


# In[7]:


aTokens = []
bTokens = []
scores = []
a =[]
b =[]
classes = []
for eachData in tqdm(pairs):
    # Change handling extra splitting of numeric characters (as it is not done in CustomTokenizer)
    atok = []
    btok = []
    for eachtok in CustomTokenizer(eachData[0]):
        if " " in eachtok:
            for aa in eachtok.split():
                atok.append(aa)
        else:
            atok.append(eachtok)
    for eachtok in CustomTokenizer(eachData[1]):
        if " " in eachtok:
            for bb in eachtok.split():
                btok.append(bb)
        else:
            btok.append(eachtok)
    aTokens.append(atok)
    bTokens.append(btok)
    #print (aTokens)
    #print (bTokens)
    #a.append(eachData[0].strip('"'))
    #b.append(eachData[1].strip('"'))
    a.append(eachData[0])
    b.append(eachData[1])
    score = float(eachData[2].strip("\n"))
    scores.append(score)
    classes.append(round_half_up(score))
    #input("WAIT")


# ### Handle short token list and single overlap "patient"

# In[8]:


for i,(eachA, eachB, eachScore) in tqdm(enumerate(zip(aTokens,bTokens,scores))):
    inter = set(eachA).intersection(set(eachB))
    
    if len(inter)==1:
#         print (i)
#         print (eachA,eachB,eachScore)
#         print (inter)
#         print (len(inter))
#         print(list(set(eachA) - inter))
#         print(set(eachB) - inter)
        aTokens[i] = list(set(eachA) - inter)
        bTokens[i] = list(set(eachB) - inter)
#         print (aTokens[i],bTokens[i])
#         input("WAIT")


# ### BioSentVec Model

# In[9]:


import sent2vec
from scipy.spatial import distance


# In[10]:


model_path = "data/embeddings/BioSentVec/model/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
model = sent2vec.Sent2vecModel()
try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print('model successfully loaded')


# ### Word2Vec Model

# In[11]:


import gensim.models as word2vec
if not os.path.exists('data/embeddings/pubmed_s100w10_min.bin'):
    raise ValueError("SKIP: You need to download the model")
     
w2vmodel = word2vec.KeyedVectors.load_word2vec_format('data/embeddings/pubmed_s100w10_min.bin', binary=True)


# In[12]:


w2vmodel.wmdistance("Hi","Hello")


# In[13]:


w2vmodel.init_sims(replace=True) 


# ### Get Sentence embedding using the BioSentVec

# In[14]:


arrayA = np.empty((0,700))
arrayB = np.empty((0,700))
index = []
cosineSentSim = []
euclideanSentSim = []
sqeuclideanSentSim = []
correlationSentSim = []
cityblockSentSim = []
wmdw2v = []
for idx, dataTokens in enumerate(zip(aTokens, bTokens)):
    index.append(idx)
    tokenizedA = " ".join(dataTokens[0])
    tokenizedB = " ".join(dataTokens[1])
    #print (tokenizedA,"\n",tokenizedB)
    sentenceVectorA = model.embed_sentence(tokenizedA)
    sentenceVectorB = model.embed_sentence(tokenizedB)
    arrayA = np.append(arrayA, sentenceVectorA,axis=0)
    arrayB = np.append(arrayB, sentenceVectorB,axis=0)
    #print (arrayA.shape, arrayB.shape)
    
    
    cosineSim = 1 - distance.cosine(sentenceVectorA, sentenceVectorB)
    cosineSentSim.append(cosineSim)
    euclideanSim = distance.euclidean(sentenceVectorA, sentenceVectorB)
    euclideanSentSim.append(euclideanSim)
    sqeuclideanSim = distance.sqeuclidean(sentenceVectorA, sentenceVectorB)
    sqeuclideanSentSim.append(sqeuclideanSim)
    corrSim = distance.correlation(sentenceVectorA, sentenceVectorB)
    correlationSentSim.append(corrSim)
    blockSim = distance.cityblock(sentenceVectorA, sentenceVectorB)
    cityblockSentSim.append(blockSim)
    
    wmd = w2vmodel.wmdistance(tokenizedA,tokenizedB)
    wmdw2v.append(wmd)
    #print('eu similarity:', euclideanSim)

    #input("WAIT")


# In[15]:


print(len(a),len(b),len(aTokens),len(bTokens),len(scores), len(classes), len(cosineSentSim), len(index), len(euclideanSentSim), len(sqeuclideanSentSim), len(correlationSentSim), len(cityblockSentSim))


# In[16]:


df = pd.DataFrame({'id':index,'a':a,'b':b,'scores':scores,'classes':classes,'aTokens':aTokens, 'bTokens':bTokens,'CosineSentSim':cosineSentSim,
                   'euSentSim':euclideanSentSim, 'sqeuSentSim':sqeuclideanSentSim, 'corrSentSim':correlationSentSim,
                   'cityblockSentSim':cityblockSentSim,'wmdw2v':wmdw2v})


# In[17]:


df.head()


# In[18]:


df['invEuSentSim'] = df.euSentSim.max() - df.euSentSim
df['invSqeuSentSim'] = df.sqeuSentSim.max() - df.sqeuSentSim


# In[19]:


df.head(10)


# ### Export the sentence embeddings

# In[20]:


print(len(index),arrayA.shape,arrayB.shape)


# In[21]:


import pickle
f = open("src/data/BioSentEmbeddings.pkl","wb")
pickle.dump([arrayA,arrayB],f)
f.close()


# ### Loading the Pickle

# In[22]:


emb = pickle.load(open("src/data/BioSentEmbeddings.pkl","rb"))


# In[23]:


emb[0].shape 


# In[24]:


df.classes.unique()


# ### Similarity Features 

# In[25]:


def sentence(tokens):
  return " ".join(tokens)

df['Sequence1'] = df['aTokens'].apply(sentence)
df['Sequence2'] = df['bTokens'].apply(sentence)
df.head()


# In[26]:


get_ipython().system('pip install py_stringmatching')
import py_stringmatching as sm


# # Token Based Similarities

# In[27]:


jac = sm.Jaccard()
df['Jaccard'] = df.apply(lambda x: jac.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[28]:


jaro = sm.Jaro()

# !pip install pyjarowinkler
# from pyjarowinkler import distance
# def jaro_similarity(word1, word2):
#   return distance.get_jaro_distance(word1, word2, winkler=False, scaling=0.1)
  
def jaccard_similarity_general(tokens1, tokens2):
  intersection = []
  for token1 in list(set(tokens1)):
    for token2 in list(set(tokens2)):
      if jaro.get_sim_score(token1, token2) > 0.7:
        if token1 not in intersection:
          intersection.append(token1)
        if token2 not in intersection:
          intersection.append(token2)
  union = set(tokens1).union(set(tokens2))
  return len(intersection)/len(union)

df['Jaccard_G'] = df.apply(lambda x: jaccard_similarity_general(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[29]:


def qgram(tokens, q):
  qgrams = []
  fix = []
  j = q
  while True:
    j -= 1
    if j == 0:
      break
    fix.append("#")
  tokens = fix + tokens + fix
  for i, token in enumerate(tokens[:-q+1]):
    qgrams.append(' '.join(tokens[i:i+q]))
  return qgrams

def qgram2(tokens):
  return qgram(tokens, 2)

def qgram3(tokens):
  return qgram(tokens, 3)

def qgram4(tokens):
  return qgram(tokens, 4)

df['Q-gram_2_Tokens1'] = df['aTokens'].apply(qgram2)
df['Q-gram_3_Tokens1'] = df['aTokens'].apply(qgram3)
df['Q-gram_4_Tokens1'] = df['aTokens'].apply(qgram4)

df['Q-gram_2_Tokens2'] = df['bTokens'].apply(qgram2)
df['Q-gram_3_Tokens2'] = df['bTokens'].apply(qgram3)
df['Q-gram_4_Tokens2'] = df['bTokens'].apply(qgram4)

df.head()


# In[30]:


df['Q2'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_2_Tokens1'], x['Q-gram_2_Tokens2']), axis=1)
df['Q3'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_3_Tokens1'], x['Q-gram_3_Tokens2']), axis=1)
df['Q4'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_4_Tokens1'], x['Q-gram_4_Tokens2']), axis=1)
df.head()


# In[31]:


cos = sm.Cosine()
df['Cosine'] = df.apply(lambda x: cos.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[32]:


dice = sm.Dice()
df['Dice'] = df.apply(lambda x: dice.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[33]:


oc = sm.OverlapCoefficient()
df['Overlap'] = df.apply(lambda x: oc.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[34]:


# Set alpha beta https://en.wikipedia.org/wiki/Tversky_index
# Setting alpha beta as 0.5 is same as Dice Similarity
tvi = sm.TverskyIndex(0.3, 0.6)
df['Tversky'] = df.apply(lambda x: tvi.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[35]:


# me = sm.MongeElkan(sim_func=NeedlemanWunsch().get_raw_score)
# me = MongeElkan(sim_func=Affine().get_raw_score)
me = sm.MongeElkan()
df['MongeElkan'] = df.apply(lambda x: me.get_raw_score(x['aTokens'], x['bTokens']) if me.get_raw_score(x['aTokens'], x['bTokens'])>0.7 else 0 , axis=1)
df.head(20)


# In[36]:


corpus = []
def generate_corpus(tokens):
  corpus.append(tokens)

df['aTokens'].apply(generate_corpus)
df['bTokens'].apply(generate_corpus)
print(len(corpus))


# In[37]:


tfidf = sm.TfIdf(corpus)
df['TfIdf'] = df.apply(lambda x: tfidf.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# # Sequence Based Similarities

# In[38]:


aff = sm.Affine()
df['Affine'] = df.apply(lambda x: aff.get_raw_score(x['Sequence1'], x['Sequence2']), axis=1)
df.Affine = df.Affine.clip(lower=0)
df.head()


# In[39]:


bd = sm.BagDistance()
df['Bag'] = df.apply(lambda x: bd.get_sim_score(x['Sequence1'], x['Sequence2']), axis=1)
df.head()


# In[40]:


ed = sm.Editex()
df['Editex'] = df.apply(lambda x: ed.get_sim_score(x['Sequence1'], x['Sequence2']) if ed.get_sim_score(x['Sequence1'], x['Sequence2']) > 0.4 else 0, axis=1)
df.head()


# In[41]:


jaro = sm.Jaro()
df['Jaro'] = df.apply(lambda x: jaro.get_sim_score(x['Sequence1'], x['Sequence2']) if jaro.get_sim_score(x['Sequence1'], x['Sequence2']) > 0.5 else 0, axis=1)
df.head()


# In[42]:


lev = sm.Levenshtein()
df['Levenshtein'] = df.apply(lambda x: lev.get_sim_score(x['Sequence1'], x['Sequence2']) if lev.get_sim_score(x['Sequence1'], x['Sequence2']) > 0.5 else 0, axis=1)
df.head()


# In[43]:


nw = sm.NeedlemanWunsch()
df['NeedlemanWunsch'] = df.apply(lambda x: nw.get_raw_score(x['Sequence1'], x['Sequence2']), axis=1)
df.NeedlemanWunsch = df.NeedlemanWunsch.clip(lower=0)
df.head()


# In[44]:


sw = sm.SmithWaterman()
df['SmithWaterman'] = df.apply(lambda x: sw.get_raw_score(x['Sequence1'], x['Sequence2']), axis=1)
df['SmithWaterman'] = df.SmithWaterman.clip(lower=0)
df.head()


# ### NLI Features

# In[45]:


nlipath = "features/NLI_HigherLevelFeatures"
dfTrain = pd.read_csv(os.path.join(nlipath,"train0_withNLIHigherLevelFeatures.csv"))
dfTrain.head()
dfDev = pd.read_csv(os.path.join(nlipath,"dev0_withNLIHigherLevelFeatures.csv"))
dfDev.head()
dfTest = pd.read_csv(os.path.join(nlipath,"test0_withNLIHigherLevelFeatures.csv"))
dfTest.head()


# In[46]:


print(len(dfTrain), len(dfDev),len(dfTest))


# In[47]:


nliFeatures = pd.concat([dfTrain, dfDev, dfTest], ignore_index=True)
len(nliFeatures)

nliFeatures.to_csv(open("src/data/nliFeatures.csv",'w'),sep=',')
nliFeatures.drop_duplicates(subset='index',keep = False, inplace = True)
len(nliFeatures)


# In[48]:


nliFeatures['sentence1'] = nliFeatures['sentence1'].str.replace('"','').str.replace(' ','')
nliFeatures['sentence2'] = nliFeatures['sentence2'].str.replace('"','').str.replace(' ','')
df['s1'] = df['a'].str.replace('"','').str.replace(' ','')
df['s2'] = df['b'].str.replace('"','').str.replace(' ','')
nliFeatures.head()


# In[49]:


print (len(df))
newdf = pd.merge(df,nliFeatures, how='left',left_on=['s1','s2','scores'], right_on = ['sentence1','sentence2','true_score']).fillna(0)
print (len(newdf), len(newdf['ModifiedESIM_2Class_Dissimilar'])- newdf['ModifiedESIM_2Class_Dissimilar'].count())

newdf[newdf.ModifiedESIM_2Class_Dissimilar.isnull()]
newdf.drop(['index','sentence1','sentence2','true_score'],axis=1, inplace=True)
newdf.head()


# ### Numeric Matchings

# In[50]:


dfNum = pd.read_csv("features/numbers/features_numbers2.csv")
print (len(dfNum))
dfNum.head()


# In[54]:


dfNum['Sentence1'] = dfNum['Sentence1'].str.replace('"','').str.replace(' ','')
dfNum['Sentence2'] = dfNum['Sentence2'].str.replace('"','').str.replace(' ','')
newdf['s1'] = newdf['a'].str.replace('"','').str.replace(' ','')
newdf['s2'] = newdf['b'].str.replace('"','').str.replace(' ','')
#nliFeatures.head(20)


# In[59]:


newdfNum = pd.merge(newdf,dfNum, how='left',left_on=['s1','s2','scores'], right_on = ['Sentence1','Sentence2','Score'])
print (len(newdfNum), len(newdfNum['Number_similarity'])- newdfNum['Number_similarity'].count())

newdfNum[newdfNum.Number_similarity.isnull()]
newdfNum.drop([ 'Sentence1', 'Sentence2','Unnamed: 0', 'Unnamed: 0.1','Numbers1','Numbers2','NumberWords1','NumberWords2','Score','String1','String2','Tokens1','Tokens2'],axis=1, inplace=True)
print ( newdfNum.columns )
newdfNum.head()


# In[65]:


len(newdf[newdfNum.id==newdfNum.Index])


# ### Get the core 

# In[66]:


coredf = newdf[["id","a","b","scores","classes","aTokens",'s1', 's2']]


# In[67]:


coredf.to_csv(open("src/data/core.csv","w"),sep=",")


# ### Domain Level Features

# In[68]:


file = os.path.join("features/domain","features_domain.csv")
domFeatures = pd.read_csv(file)
print (len(domFeatures))
domFeatures.head()


# In[81]:


newDomDf = pd.merge(coredf,domFeatures,how="left",left_on=["id"],right_on=["Index"])
newDomDf.head()
newDomDf.columns
#len(newDomDf["Jaccard"]) - len(newDomDf["Jaccard"].count())
#newDomDf = newDomDf["id","Jaro","Levenshtein","NeedlemanWunsch","SmithWaterman"]


# In[86]:


newDomDf =newDomDf[['id','Jaccard', 'Jaccard_G', 'Q2', 'Q3', 'Q4', 'Cosine',
       'Dice', 'Overlap', 'Tversky', 'MongeElkan', 'TfIdf', 'Affine', 'Bag', 'Editex', 'Jaro', 'Levenshtein',
       'NeedlemanWunsch', 'SmithWaterman']]


# In[87]:


len(newDomDf)
newDomDf.to_csv(open("src/data/domain.csv",'w'),sep=',')


# ### Domain Embedding

# In[88]:


file = os.path.join("features/domain","features_domain_embeddings.csv")
domFeatures = pd.read_csv(file)
print (len(domFeatures))
domFeatures.head()


# In[81]:


newDomDf = pd.merge(coredf,domFeatures,how="left",left_on=["id"],right_on=["Index"])
newDomDf.head()
newDomDf.columns
#len(newDomDf["Jaccard"]) - len(newDomDf["Jaccard"].count())
#newDomDf = newDomDf["id","Jaro","Levenshtein","NeedlemanWunsch","SmithWaterman"]


# In[86]:


newDomDf =newDomDf[['id','Jaccard', 'Jaccard_G', 'Q2', 'Q3', 'Q4', 'Cosine',
       'Dice', 'Overlap', 'Tversky', 'MongeElkan', 'TfIdf', 'Affine', 'Bag', 'Editex', 'Jaro', 'Levenshtein',
       'NeedlemanWunsch', 'SmithWaterman']]


# In[87]:


len(newDomDf)
newDomDf.to_csv(open("src/data/domain.csv",'w'),sep=',')


# ### The following data is used to quickly establish the performance of models (NOT THE FINAL CODE)

# In[53]:


newdfNum.to_csv(open("src/data/allfeatures.csv",'w'),sep=',')


# In[54]:


dt = pd.read_csv("src/data/allfeatures.csv")
                   


# In[55]:


dt.classes.values.shape


# In[ ]:





# In[ ]:




