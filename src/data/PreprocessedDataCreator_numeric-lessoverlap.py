#!/usr/bin/env python
# coding: utf-8

# ## Analysis and Validation of our Custom Tokenizer

# ### Why ? Issues noticed - 
# #### 1. Drug names in [] missing which is essential (added a new function in tokenizer)
# #### 2. numeric entries converted to a string of tokens (Handling now)
# #### 3. extra token 'point' for period at the end of statements (fixed and removed)

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
os.chdir('homepath')
print (os.getcwd())
# pathFeatures = 'models/token+sequence'
# fileFeatures = os.path.join(pathFeatures,'df.csv')


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
    a.append(eachData[0].strip('"'))
    b.append(eachData[1].strip('"'))
    score = float(eachData[2].strip("\n"))
    scores.append(score)
    classes.append(round_half_up(score))
    #input("WAIT")


# ### BioSentVec Model

# In[8]:


import sent2vec
from scipy.spatial import distance


# In[ ]:


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
df['MongeElkan'] = df.apply(lambda x: me.get_raw_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


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
df.head()


# In[39]:


bd = sm.BagDistance()
df['Bag'] = df.apply(lambda x: bd.get_sim_score(x['Sequence1'], x['Sequence2']), axis=1)
df.head()


# In[40]:


ed = sm.Editex()
df['Editex'] = df.apply(lambda x: ed.get_sim_score(x['Sequence1'], x['Sequence2']), axis=1)
df.head()


# In[41]:


jaro = sm.Jaro()
df['Jaro'] = df.apply(lambda x: jaro.get_sim_score(x['Sequence1'], x['Sequence2']), axis=1)
df.head()


# In[42]:


lev = sm.Levenshtein()
df['Levenshtein'] = df.apply(lambda x: lev.get_sim_score(x['Sequence1'], x['Sequence2']), axis=1)
df.head()


# In[43]:


nw = sm.NeedlemanWunsch()
df['NeedlemanWunsch'] = df.apply(lambda x: nw.get_raw_score(x['Sequence1'], x['Sequence2']), axis=1)
df.head()


# In[44]:


sw = sm.SmithWaterman()
df['SmithWaterman'] = df.apply(lambda x: sw.get_raw_score(x['Sequence1'], x['Sequence2']), axis=1)
df.head()


# ### The following data is used to quickly establish the performance of models (NOT THE FINAL CODE)

# In[45]:


df.to_csv(open("src/data/TokenizedNewData.csv",'w'),sep=',')


# In[46]:


dt = pd.read_csv("src/data/TokenizedNewData.csv")
                   


# In[47]:


dt.classes.values.shape


# In[ ]:




