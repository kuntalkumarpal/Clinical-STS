#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm


# In[3]:


### Set up root directory
print (os.getcwd())
os.chdir('homepath')
print (os.getcwd())


# In[3]:


tstdataFile = 'srcdata'


# ### 1. Source data processing

# In[4]:


import src.data.DataLoader as CustomDataLoader
from src.features.CustomTokenizer import CustomTokenizer


# In[5]:


pairs = CustomDataLoader.DataLoader(tstdataFile)


# In[6]:


numOriginal = len(pairs)


# In[1]:


numOriginal


# In[7]:


pairs[0]


# ### Generated Data Loader

# In[8]:


isGen = False
isCovertionToTsvReq = True


# In[9]:


if isGen:
    if isCovertionToTsvReq :
        genCsvFile = "data/train/GeneratedAbove4.csv"
        genTsvFile = "data/train/GeneratedAbove4.tsv"
        
        import csv
        i=0
        with open(genCsvFile,'r') as csvin, open(genTsvFile, 'w') as tsvout:

            csvin = csv.reader(csvin)
            tsvout = csv.writer(tsvout, delimiter='\t')

            for row in csvin:
                i+=1
                if i>1: 
                    tsvout.writerow(row)
    
    genData = CustomDataLoader.DataLoader(genTsvFile)
    print (len(genData))
    print (genData[8])
    
    pairs = pairs + genData
    
    print (len(pairs))
    print (pairs[0])
    print (pairs[1650])


# ### Parsing and Tokenization

# In[10]:


def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return int(np.floor(n*multiplier + 0.5) / multiplier)


# In[11]:


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
    a.append(eachData[0])
    b.append(eachData[1])
    score = float(eachData[2].strip("\n"))
    scores.append(score)
    classes.append(round_half_up(score))


# ### Handle short token list and single overlap "patient"

# In[12]:


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

# In[13]:


import sent2vec
from scipy.spatial import distance


# In[14]:


model_path = "data/embeddings/BioSentVec/model/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
model = sent2vec.Sent2vecModel()
try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print('model successfully loaded')


# ### Word2Vec Model

# In[15]:


import gensim.models as word2vec
if not os.path.exists('data/embeddings/pubmed_s100w10_min.bin'):
    raise ValueError("SKIP: You need to download the model")
     
w2vmodel = word2vec.KeyedVectors.load_word2vec_format('data/embeddings/pubmed_s100w10_min.bin', binary=True)


# ### Get Sentence embedding using the BioSentVec

# In[16]:


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

    #input("WAIT")


# In[17]:


# if isGen:
#     df = pd.DataFrame({'id':index,'a':a,'b':b,'scores':scores,'classes':classes,'aTokens':aTokens, 'bTokens':bTokens,'CosineSentSim':cosineSentSim,
#                    'euSentSim':euclideanSentSim, 'sqeuSentSim':sqeuclideanSentSim, 'corrSentSim':correlationSentSim,
#                    'cityblockSentSim':cityblockSentSim,'wmdw2v':wmdw2v})
# else:
isGeneratedData = [0]*numOriginal + [1]*(len(pairs)-numOriginal)
print (len(isGeneratedData))
df = pd.DataFrame({'id':index,'a':a,'b':b,'scores':scores,'classes':classes,'aTokens':aTokens, 'bTokens':bTokens,'CosineSentSim':cosineSentSim,
               'euSentSim':euclideanSentSim, 'sqeuSentSim':sqeuclideanSentSim, 'corrSentSim':correlationSentSim,
               'cityblockSentSim':cityblockSentSim,'wmdw2v':wmdw2v,'isGen':isGeneratedData})


# In[18]:


print (len(df))
print (df.columns)


# In[19]:


df.head()


# ### 2. Get the core data

# In[20]:


featurePath = 'src/features/FeatureProcessing'


# In[21]:


def sentence(tokens):
  return " ".join(tokens)

df['s1_join'] = df['a'].str.replace('"','').str.replace(" ","")
df['s2_join'] = df['b'].str.replace('"','').str.replace(" ","")
df['Sequence1'] =df['aTokens'].apply(sentence)
df['Sequence2'] =df['bTokens'].apply(sentence)
coredf = df[['id', 'a', 'b', 'scores', 'classes', 'aTokens', 'bTokens','s1_join','s2_join','Sequence1','Sequence2','isGen']]


# In[22]:


coredf.to_csv(open(os.path.join(featurePath,"core.csv"),"w"),sep=",",index=False)


# ### 3. Embedding Based Similarity

# In[23]:


embdf = df[['id','CosineSentSim', 'euSentSim', 'sqeuSentSim', 'corrSentSim',
       'cityblockSentSim', 'wmdw2v']]


# In[24]:


embdf.to_csv(open(os.path.join(featurePath,"embedding.csv"),"w"),sep=",",index=False)


# ### Export the sentence embeddings

# In[25]:


print(len(index),arrayA.shape,arrayB.shape)


# In[26]:


import pickle
f = open(os.path.join(featurePath,"BioSentEmbeddings.pkl"),"wb")
pickle.dump([arrayA,arrayB],f)
f.close()


# ### Loading the Pickle and Test

# In[27]:


emb = pickle.load(open(os.path.join(featurePath,"BioSentEmbeddings.pkl"),"rb"))


# In[28]:


emb[0].shape 


# ### 4. String Based Similarity

# In[29]:


# df = pd.read_csv(os.path.join(featurePath,"core.csv"))
# df.head()


# In[30]:


get_ipython().system('pip install py_stringmatching')
import py_stringmatching as sm


# # Token Based Similarities

# In[31]:


jac = sm.Jaccard()
#df['aTokens']
df['Jaccard'] = df.apply(lambda x: jac.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[32]:


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


# In[33]:


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


# In[34]:


df['Q2'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_2_Tokens1'], x['Q-gram_2_Tokens2']), axis=1)
df['Q3'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_3_Tokens1'], x['Q-gram_3_Tokens2']), axis=1)
df['Q4'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_4_Tokens1'], x['Q-gram_4_Tokens2']), axis=1)
df.head()


# In[35]:


cos = sm.Cosine()
df['Cosine'] = df.apply(lambda x: cos.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[36]:


dice = sm.Dice()
df['Dice'] = df.apply(lambda x: dice.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[37]:


oc = sm.OverlapCoefficient()
df['Overlap'] = df.apply(lambda x: oc.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[38]:


# Set alpha beta https://en.wikipedia.org/wiki/Tversky_index
# Setting alpha beta as 0.5 is same as Dice Similarity
tvi = sm.TverskyIndex(0.3, 0.6)
df['Tversky'] = df.apply(lambda x: tvi.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# In[39]:


# me = sm.MongeElkan(sim_func=NeedlemanWunsch().get_raw_score)
# me = MongeElkan(sim_func=Affine().get_raw_score)
me = sm.MongeElkan()
df['MongeElkan'] = df.apply(lambda x: me.get_raw_score(x['aTokens'], x['bTokens']) if me.get_raw_score(x['aTokens'], x['bTokens'])>0.7 else 0 , axis=1)
df.head(20)


# In[40]:


corpus = []
def generate_corpus(tokens):
  corpus.append(tokens)

df['aTokens'].apply(generate_corpus)
df['bTokens'].apply(generate_corpus)
print(len(corpus))


# In[41]:


tfidf = sm.TfIdf(corpus)
df['TfIdf'] = df.apply(lambda x: tfidf.get_sim_score(x['aTokens'], x['bTokens']), axis=1)
df.head()


# # Sequence Based Similarities

# In[42]:


aff = sm.Affine()
df['Affine'] = df.apply(lambda x: aff.get_raw_score(x['Sequence1'], x['Sequence2']), axis=1)
df.Affine = df.Affine.clip(lower=0)
df.head()


# In[43]:


bd = sm.BagDistance()
df['Bag'] = df.apply(lambda x: bd.get_sim_score(x['Sequence1'], x['Sequence2']), axis=1)
df.head()


# In[44]:


ed = sm.Editex()
df['Editex'] = df.apply(lambda x: ed.get_sim_score(x['Sequence1'], x['Sequence2']) if ed.get_sim_score(x['Sequence1'], x['Sequence2']) > 0.4 else 0, axis=1)
df.head()


# In[45]:


jaro = sm.Jaro()
df['Jaro'] = df.apply(lambda x: jaro.get_sim_score(x['Sequence1'], x['Sequence2']) if jaro.get_sim_score(x['Sequence1'], x['Sequence2']) > 0.5 else 0, axis=1)
df.head()


# In[46]:


lev = sm.Levenshtein()
df['Levenshtein'] = df.apply(lambda x: lev.get_sim_score(x['Sequence1'], x['Sequence2']) if lev.get_sim_score(x['Sequence1'], x['Sequence2']) > 0.5 else 0, axis=1)
df.head()


# In[47]:


nw = sm.NeedlemanWunsch()
df['NeedlemanWunsch'] = df.apply(lambda x: nw.get_raw_score(x['Sequence1'], x['Sequence2']), axis=1)
df.NeedlemanWunsch = df.NeedlemanWunsch.clip(lower=0)
df.head()


# In[48]:


sw = sm.SmithWaterman()
df['SmithWaterman'] = df.apply(lambda x: sw.get_raw_score(x['Sequence1'], x['Sequence2']), axis=1)
df['SmithWaterman'] = df.SmithWaterman.clip(lower=0)
df.head()


# In[49]:


df.columns


# In[50]:


string = df[['id','Jaccard', 'Jaccard_G', 'Q2', 'Q3', 'Q4', 'Cosine',
       'Dice', 'Overlap', 'Tversky', 'MongeElkan', 'TfIdf', 'Affine','Bag',
       'Editex', 'Jaro', 'Levenshtein', 'NeedlemanWunsch', 'SmithWaterman']]


# In[51]:


string.to_csv(os.path.join(featurePath,"string.csv"),sep=",",index=False)


# In[52]:


df.drop(['CosineSentSim', 'euSentSim', 'sqeuSentSim', 'corrSentSim',
       'cityblockSentSim', 'wmdw2v','Jaccard', 'Jaccard_G', 'Q-gram_2_Tokens1',
       'Q-gram_3_Tokens1', 'Q-gram_4_Tokens1', 'Q-gram_2_Tokens2',
       'Q-gram_3_Tokens2', 'Q-gram_4_Tokens2', 'Q2', 'Q3', 'Q4', 'Cosine',
       'Dice', 'Overlap', 'Tversky', 'MongeElkan', 'TfIdf', 'Affine', 'Bag',
       'Editex', 'Jaro', 'Levenshtein', 'NeedlemanWunsch', 'SmithWaterman'],axis=1, inplace=True)


# ### 5. NLI Features

# In[53]:


nlipath = "features/NLI_HigherLevelFeatures2"
dfTrain = pd.read_csv(os.path.join(nlipath,"train0_withNLIHigherLevelFeatures.csv"))
dfTrain.head()
dfDev = pd.read_csv(os.path.join(nlipath,"dev0_withNLIHigherLevelFeatures.csv"))
dfDev.head()
dfTest = pd.read_csv(os.path.join(nlipath,"test0_withNLIHigherLevelFeatures.csv"))
dfTest.head()


# In[54]:


print(len(dfTrain), len(dfDev),len(dfTest))


# In[55]:


nliFeatures = pd.concat([dfTrain, dfDev, dfTest], ignore_index=True)
#nliFeatures.to_csv(open("src/data/nliFeatures.csv",'w'),sep=',')
nliFeatures.drop_duplicates(subset='index',keep = False, inplace = True)
len(nliFeatures)


# In[56]:


nliFeatures['sentence1'] = nliFeatures['sentence1'].str.replace('"','').str.replace(' ','')
nliFeatures['sentence2'] = nliFeatures['sentence2'].str.replace('"','').str.replace(' ','')
# df['s1'] = df['a'].str.replace('"','').str.replace(' ','')
# df['s2'] = df['b'].str.replace('"','').str.replace(' ','')
nliFeatures.head()


# In[57]:


newdf = pd.merge(df,nliFeatures, how='left',left_on=['s1_join','s2_join','scores'], right_on = ['sentence1','sentence2','true_score']).fillna(0)
print (len(newdf), len(newdf['ModifiedESIM_2Class_Dissimilar'])- newdf['ModifiedESIM_2Class_Dissimilar'].count())

newdf[newdf.ModifiedESIM_2Class_Dissimilar.isnull()]
newdf.drop(['index','sentence1','sentence2','true_score'],axis=1, inplace=True)
newdf.head()
newdf.columns


# In[58]:


nlidf = newdf[['id','ModifiedESIM_2Class_Dissimilar',
       'ModifiedESIM_2Class_Similar', 'ModifiedESIM_3Class_Dissimilar',
       'ModifiedESIM_3Class_NDNS', 'ModifiedESIM_3Class_Similar',
       'ModifiedESIM_p2h_h2p_2Class_Dissimilar',
       'ModifiedESIM_p2h_h2p_2Class_Similar',
       'ModifiedESIM_p2h_h2p_3Class_Dissimilar',
       'ModifiedESIM_p2h_h2p_3Class_NDNS',
       'ModifiedESIM_p2h_h2p_3Class_Similar', 'OriginalESIM_2Class_Dissimilar',
       'OriginalESIM_2Class_Similar', 'OriginalESIM_3Class_Dissimilar',
       'OriginalESIM_3Class_NDNS', 'OriginalESIM_3Class_Similar',
       'OriginalESIM_p2h_h2p_2Class_Dissimilar',
       'OriginalESIM_p2h_h2p_2Class_Similar',
       'OriginalESIM_p2h_h2p_3Class_Dissimilar',
       'OriginalESIM_p2h_h2p_3Class_NDNS',
       'OriginalESIM_p2h_h2p_3Class_Similar']]


# In[59]:


nlidf.to_csv(os.path.join(featurePath,"nli.csv"),sep=",",index=False)


# In[60]:


df.columns


# ### 6. Numeric Matchings

# In[61]:


import re
def numbers(s):
  return re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)

df['Numbers1'] = df.apply(lambda x: numbers(x['Sequence1']), axis=1)
df['Numbers2'] = df.apply(lambda x: numbers(x['Sequence2']), axis=1)
df.head()


# In[62]:


get_ipython().system('pip install inflect')
import inflect
ie = inflect.engine()


# In[63]:


def numberwords(nums):
  if len(nums) == 0:
    return ""
  out = ""
  for num in nums:
    out += ie.number_to_words(num) + " "
  return out

df['NumberWords1'] = df.apply(lambda x: numberwords(x['Numbers1']), axis=1)
df['NumberWords2'] = df.apply(lambda x: numberwords(x['Numbers2']), axis=1)
df.head()


# ### BioWordVec Model 

# In[64]:


from gensim.models import KeyedVectors
import time 

if not os.path.exists('data/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin'):
    raise ValueError("SKIP: You need to download the model")

stime = time.time()     
wv = KeyedVectors.load_word2vec_format("data/embeddings/BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True)
print (time.time() - stime)


# In[65]:


wv.wmdistance('one', 'once')


# In[66]:


df.replace(np.nan, '', regex=True)

def similarity(a, b):
    if len(a) == 0 and len(b) == 0:
        return 0
    if len(a) == 0:
        return np.NaN
    if len(b) == 0:
        return np.NaN
    return wv.wmdistance(a, b)


df['Number_similarity'] = df.apply(lambda x: similarity(x['NumberWords1'], x['NumberWords2']), axis=1)
df.head()


# In[67]:


df.Number_similarity = df.Number_similarity.fillna(round(df.Number_similarity.max()+1))


# ### 7. Word Mover Distance String + Domain

# In[68]:


df['wordmover'] = df.apply(lambda x: similarity(x['Sequence1'], x['Sequence2']), axis=1)


# In[69]:


df.wordmover = df.wordmover.fillna(round(df.wordmover.max()+1))


# In[70]:


df.columns


# In[71]:


wordmoverdf = df[['id','Number_similarity', 'wordmover']]


# In[72]:


wordmoverdf.to_csv(os.path.join(featurePath,"wordmover.csv"),sep=",",index=False)


# In[73]:


df.drop([ 'Numbers1', 'Numbers2','NumberWords1', 'NumberWords2', 'Number_similarity', 'wordmover'],axis=1, inplace=True)


# ### 8. Domain

# In[74]:


import os

def prefix(a, b):
  paths = [a, b]
  prefix = os.path.commonprefix(paths)
  while True:
    if len(prefix) == 0 or not prefix[-1].isalpha():
      break
    else:
      prefix = prefix[:-1]
  return prefix

df['prefix'] = df.apply(lambda x: prefix(x['a'], x['b']), axis=1)
df.head()


# In[75]:


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

df['Domain1'] = df.apply(lambda x: remove_prefix(x['Sequence1'], x['prefix']), axis=1)
df['Domain2'] = df.apply(lambda x: remove_prefix(x['Sequence2'], x['prefix']), axis=1)
df.head()


# In[76]:


get_ipython().system('pip install nltk')
import nltk
import src.features.CustomTokenizer as ct

def token(sent):
  words = nltk.word_tokenize(sent)
  words = ct.replace_numbers(words)
  words = ct.remove_non_ascii(words)
  words = ct.lemmatize_verbs(words)
  return words

df['Tokens1'] = df['Domain1'].apply(token)
df['Tokens2'] = df['Domain2'].apply(token)

df.head()


# In[77]:



get_ipython().system('pip install py_stringmatching')
import py_stringmatching as sm
from ast import literal_eval


# In[78]:


jac = sm.Jaccard()
df['DomJaccard'] = df.apply(lambda x: jac.get_sim_score(x['Tokens1'], x['Tokens2']), axis=1)
df.head()


# In[79]:


jaro = sm.Jaro()
  
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

df['DomJaccard_G'] = df.apply(lambda x: jaccard_similarity_general(x['Tokens1'], x['Tokens2']), axis=1)
df.head()


# In[80]:


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

df['Q-gram_2_Tokens1'] = df['Tokens1'].apply(qgram2)
df['Q-gram_3_Tokens1'] = df['Tokens1'].apply(qgram3)
df['Q-gram_4_Tokens1'] = df['Tokens1'].apply(qgram4)

df['Q-gram_2_Tokens2'] = df['Tokens2'].apply(qgram2)
df['Q-gram_3_Tokens2'] = df['Tokens2'].apply(qgram3)
df['Q-gram_4_Tokens2'] = df['Tokens2'].apply(qgram4)

df.head()


# In[81]:


df['DomQ2'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_2_Tokens1'], x['Q-gram_2_Tokens2']), axis=1)
df['DomQ3'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_3_Tokens1'], x['Q-gram_3_Tokens2']), axis=1)
df['DomQ4'] = df.apply(lambda x: jac.get_sim_score(x['Q-gram_4_Tokens1'], x['Q-gram_4_Tokens2']), axis=1)
df.head()


# In[82]:


cos = sm.Cosine()
df['DomCosine'] = df.apply(lambda x: cos.get_sim_score(x['Tokens1'], x['Tokens2']), axis=1)
df.head()


# In[83]:


dice = sm.Dice()
df['DomDice'] = df.apply(lambda x: dice.get_sim_score(x['Tokens1'], x['Tokens2']), axis=1)
df.head()


# In[84]:


oc = sm.OverlapCoefficient()
df['DomOverlap'] = df.apply(lambda x: oc.get_sim_score(x['Tokens1'], x['Tokens2']), axis=1)
df.head()


# In[85]:


# Set alpha beta https://en.wikipedia.org/wiki/Tversky_index
# Setting alpha beta as 0.5 is same as Dice Similarity
tvi = sm.TverskyIndex(0.3, 0.6)
df['DomTversky'] = df.apply(lambda x: tvi.get_sim_score(x['Tokens1'], x['Tokens2']), axis=1)
df.head()


# In[86]:


# me = sm.MongeElkan(sim_func=NeedlemanWunsch().get_raw_score)
# me = MongeElkan(sim_func=Affine().get_raw_score)
me = sm.MongeElkan()
df['DomMongeElkan'] = df.apply(lambda x: me.get_raw_score(x['Tokens1'], x['Tokens2']), axis=1)
df.head()


# In[87]:


corpus = []
def generate_corpus(tokens):
  corpus.append(tokens)

df['Tokens1'].apply(generate_corpus)
df['Tokens2'].apply(generate_corpus)
print(len(corpus))


# In[88]:


tfidf = sm.TfIdf(corpus)
df['DomTfIdf'] = df.apply(lambda x: tfidf.get_sim_score(x['Tokens1'], x['Tokens2']), axis=1)
df.head()


# In[89]:


df['DomSequence1'] = df.apply(lambda x: ' '.join(x['Tokens1']), axis=1)
df['DomSequence2'] = df.apply(lambda x: ' '.join(x['Tokens2']), axis=1)
df.head()


# In[90]:


aff = sm.Affine()
df['DomAffine'] = df.apply(lambda x: aff.get_raw_score(x['DomSequence1'], x['DomSequence2']), axis=1)
df.head()


# In[91]:


bd = sm.BagDistance()
df['DomBag'] = df.apply(lambda x: bd.get_sim_score(x['DomSequence1'], x['DomSequence2']), axis=1)
df.head()


# In[92]:


ed = sm.Editex()
df['DomEditex'] = df.apply(lambda x: ed.get_sim_score(x['DomSequence1'], x['DomSequence2']), axis=1)
df.head()


# In[93]:


jaro = sm.Jaro()
df['DomJaro'] = df.apply(lambda x: jaro.get_sim_score(x['DomSequence1'], x['DomSequence2']), axis=1)
df.head()


# In[94]:


lev = sm.Levenshtein()
df['DomLevenshtein'] = df.apply(lambda x: lev.get_sim_score(x['DomSequence1'], x['DomSequence2']), axis=1)
df.head()


# In[95]:


nw = sm.NeedlemanWunsch()
df['DomNeedlemanWunsch'] = df.apply(lambda x: nw.get_raw_score(x['DomSequence1'], x['DomSequence2']), axis=1)
df.head()


# In[96]:


sw = sm.SmithWaterman()
df['DomSmithWaterman'] = df.apply(lambda x: sw.get_raw_score(x['DomSequence1'], x['DomSequence2']), axis=1)
df.head()


# In[97]:


df['DomWordmover'] = df.apply(lambda x: similarity(x['Domain1'], x['Domain2']), axis=1)


# In[98]:


df.DomWordmover = df.DomWordmover.fillna(round(df.DomWordmover.max()+1))


# In[99]:


df.columns


# In[100]:


domstringdf = df[['id','DomJaccard', 'DomJaccard_G', 'DomQ2', 'DomQ3', 'DomQ4',
       'DomCosine', 'DomDice', 'DomOverlap', 'DomTversky', 'DomMongeElkan',
       'DomTfIdf', 'DomSequence1', 'DomSequence2', 'DomAffine', 'DomBag',
       'DomEditex', 'DomJaro', 'DomLevenshtein', 'DomNeedlemanWunsch',
       'DomSmithWaterman','DomWordmover']]


# In[101]:


domstringdf.to_csv(os.path.join(featurePath,"domstring.csv"),sep=",",index=False)


# In[102]:


df.drop([ 'DomJaccard', 'DomJaccard_G', 'Q-gram_2_Tokens1',
       'Q-gram_3_Tokens1', 'Q-gram_4_Tokens1', 'Q-gram_2_Tokens2',
       'Q-gram_3_Tokens2', 'Q-gram_4_Tokens2', 'DomQ2', 'DomQ3', 'DomQ4',
       'DomCosine', 'DomDice', 'DomOverlap', 'DomTversky', 'DomMongeElkan',
       'DomTfIdf', 'DomAffine', 'DomBag',
       'DomEditex', 'DomJaro', 'DomLevenshtein', 'DomNeedlemanWunsch',
       'DomSmithWaterman', 'DomWordmover'],axis=1,inplace=True)


# In[103]:


df.head()


# ### Domain Embedding Based (BioSenteVec)

# In[104]:


df['DomA_BioSentVec'] = df.apply(lambda x: model.embed_sentence(x['DomSequence1']), axis=1)
df['DomB_BioSentVec'] = df.apply(lambda x: model.embed_sentence(x['DomSequence2']), axis=1)
df.head()


# In[105]:


df['DomEmbCosine'] = df.apply(lambda x: 1- distance.cosine(x['DomA_BioSentVec'],x['DomB_BioSentVec']), axis=1)
df['DomEmbEuclidean'] = df.apply(lambda x: distance.euclidean(x['DomA_BioSentVec'],x['DomB_BioSentVec']), axis=1)
df['DomEmbSqEuclidean'] = df.apply(lambda x: distance.sqeuclidean(x['DomA_BioSentVec'],x['DomB_BioSentVec']), axis=1)
df['DomEmbCorrelation'] = df.apply(lambda x: distance.correlation(x['DomA_BioSentVec'],x['DomB_BioSentVec']), axis=1)
df['DomEmbCityblock'] = df.apply(lambda x: distance.cityblock(x['DomA_BioSentVec'],x['DomB_BioSentVec']), axis=1)

df.head()


# In[106]:


df.columns


# In[107]:


domembeddf = df[['id','DomA_BioSentVec',
       'DomB_BioSentVec', 'DomEmbCosine', 'DomEmbEuclidean',
       'DomEmbSqEuclidean', 'DomEmbCorrelation', 'DomEmbCityblock']]


# In[108]:


domembeddf.to_csv(os.path.join(featurePath,"domEmbedding.csv"),sep=",",index=False)


# ### AVG & MAX based word embeddings BioWordVec

# In[108]:


# df.drop(['prefix', 'Domain1',
#        'Domain2', 'Tokens1', 'Tokens2', 'DomSequence1', 'DomSequence2',
#        'DomA_BioSentVec', 'DomB_BioSentVec', 'DomEmbCosine', 'DomEmbEuclidean',
#        'DomEmbSqEuclidean', 'DomEmbCorrelation', 'DomEmbCityblock'],axis=1,inplace=True)
# df.head()


# In[109]:


# import spacy  
# sp = spacy.load('en_core_web_sm')


# In[226]:


# def getPos(sentence, pos):
#   out = []
#   for word in sp(sentence):
#     if word.pos_ == pos:
#       out.append(word.text.lower())
#   return out


# In[227]:


# df['Noun1'] = df.apply(lambda x: getPos(x['Sequence1'], "NOUN"), axis=1)
# df['Noun2'] = df.apply(lambda x: getPos(x['Sequence2'], "NOUN"), axis=1)
# df.head()


# In[228]:


# df['Verb1'] = df.apply(lambda x: getPos(x['Sequence1'], "VERB"), axis=1)
# df['Verb2'] = df.apply(lambda x: getPos(x['Sequence2'], "VERB"), axis=1)
# df.head()


# In[229]:


# df['Adjective1'] = df.apply(lambda x: getPos(x['Sequence1'], "ADJ"), axis=1)
# df['Adjective2'] = df.apply(lambda x: getPos(x['Sequence2'], "ADJ"), axis=1)
# df.head()


# ### Embedding max and average

# In[230]:


# def average(words):
#     if len(words) == 0:
#         return np.zeros(200)
#     w = words[0]
#     avg = 0
#     try:
#         avg = wv[w]
#     except:
#         print(w,":notfound")
#     for w in words[1:]:
#         try:
#             avg = avg + wv[w]
#         except:
#             print(w,":notfound")
#     avg = avg/len(words)
#     return avg

# def maxim(words):
#     if len(words) == 0:
#         return np.zeros(200)
#     w = words[0]
#     max=0
#     try:
#         max = wv[w]
#     except:
#         print(w,":notfound")
#     for w in words[1:]:
#         try:
#             max = np.maximum(max, wv[w])
#         except:
#             print(w,":notfound")
#     return max


# In[231]:


# df.loc[:,'n1'] = df.loc[:,'Noun1'].apply(literal_eval)
# df.loc[:,'n2'] = df.loc[:,'Noun2'].apply(literal_eval)


# In[232]:


# #df.drop(['n1','n2'],axis=1,inplace=True)
# df.head()


# In[233]:


# df['Embedding_Avg_N1'] = df.apply(lambda x: average(x['Noun1']), axis=1)
# df['Embedding_Avg_N2'] = df.apply(lambda x: average(x['Noun2']), axis=1)
# df['Embedding_Max_N1'] = df.apply(lambda x: maxim(x['Noun1']), axis=1)
# df['Embedding_Max_N2'] = df.apply(lambda x: maxim(x['Noun1']), axis=1)


# In[234]:



# df.loc[:,'Verb1'] = df.loc[:,'Verb1'].apply(literal_eval)
# df.loc[:,'Verb2'] = df.loc[:,'Verb2'].apply(literal_eval)


# In[236]:



# df['Embedding_Avg_V1'] = df.apply(lambda x: average(x['Verb1']), axis=1)
# df['Embedding_Avg_V2'] = df.apply(lambda x: average(x['Verb2']), axis=1)
# df['Embedding_Max_V1'] = df.apply(lambda x: maxim(x['Verb1']), axis=1)
# df['Embedding_Max_V2'] = df.apply(lambda x: maxim(x['Verb2']), axis=1)


# In[237]:



# df.loc[:,'Adjective1'] = df.loc[:,'Adjective1'].apply(literal_eval)
# df.loc[:,'Adjective2'] = df.loc[:,'Adjective2'].apply(literal_eval)


# In[238]:



# df['Embedding_Avg_A1'] = df.apply(lambda x: average(x['Adjective1']), axis=1)
# df['Embedding_Avg_A2'] = df.apply(lambda x: average(x['Adjective2']), axis=1)
# df['Embedding_Max_A1'] = df.apply(lambda x: maxim(x['Adjective1']), axis=1)
# df['Embedding_Max_A2'] = df.apply(lambda x: maxim(x['Adjective2']), axis=1)


# In[239]:


# df.head()


# In[240]:


# def format(str):
#   str = str[1:-1]
#   return np.fromstring(str, dtype=float, sep=" ")

# for col in ['Embedding_Avg_N1', 'Embedding_Avg_N2', 'Embedding_Avg_V1', 'Embedding_Avg_V2', 'Embedding_Avg_A1', 'Embedding_Avg_A2', 'Embedding_Max_N1', 'Embedding_Max_N2', 'Embedding_Max_V1', 'Embedding_Max_V2', 'Embedding_Max_A1', 'Embedding_Max_A2']:
#   df[col] = df.apply(lambda x: format(x[col]), axis=1)

# df.head()


# In[241]:


# def avg(n, v, a):
#     return (n + v + a) / 3


# def maxxx(n, v, a):
#   maxx = n
#   maxx = np.maximum(maxx, v)
#   maxx = np.maximum(maxx, a)
#   return maxx


# In[242]:


# df['Embedding_Avg_1'] = df.apply(lambda x: avg(x['Embedding_Avg_N1'], x['Embedding_Avg_V1'], x['Embedding_Avg_A1']), axis=1)
# df['Embedding_Avg_2'] = df.apply(lambda x: avg(x['Embedding_Avg_N2'], x['Embedding_Avg_V2'], x['Embedding_Avg_A2']), axis=1)

# df['Embedding_Max_1'] = df.apply(lambda x: avg(x['Embedding_Max_N1'], x['Embedding_Max_V1'], x['Embedding_Max_A1']), axis=1)
# df['Embedding_Max_2'] = df.apply(lambda x: avg(x['Embedding_Max_N2'], x['Embedding_Max_V2'], x['Embedding_Max_A2']), axis=1)


# In[243]:


# from numpy.linalg import norm
# from scipy.spatial import distance
# import math

# # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
# def cosine(a, b):
#   if norm(a) == 0 or norm(b) == 0:
#     return 0
#   return 1 - distance.cosine(a, b)


# df['Cosine_Avg'] = df.apply(lambda x: cosine(x['Embedding_Avg_1'], x['Embedding_Avg_2']), axis=1)
# df['Cosine_Max'] = df.apply(lambda x: cosine(x['Embedding_Max_1'], x['Embedding_Max_2']), axis=1)
# df['Cosine_N_Avg'] = df.apply(lambda x: cosine(x['Embedding_Avg_N1'], x['Embedding_Avg_N2']), axis=1)
# df['Cosine_N_Max'] = df.apply(lambda x: cosine(x['Embedding_Max_N1'], x['Embedding_Max_N2']), axis=1)
# df.head()


# In[244]:


# df.columns


# In[245]:


# df.head(20)


# In[246]:


# posembdf = df['id', 'Embedding_Avg_N1',
#        'Embedding_Avg_N2', 'Embedding_Max_N1', 'Embedding_Max_N2',
#        'Embedding_Avg_V1', 'Embedding_Avg_V2', 'Embedding_Max_V1',
#        'Embedding_Max_V2', 'Embedding_Avg_A1', 'Embedding_Avg_A2',
#        'Embedding_Max_A1', 'Embedding_Max_A2', 'Embedding_Avg_1',
#        'Embedding_Avg_2', 'Embedding_Max_1', 'Embedding_Max_2', 'Cosine_Avg',
#        'Cosine_Max', 'Cosine_N_Avg', 'Cosine_N_Max']


# In[ ]:




