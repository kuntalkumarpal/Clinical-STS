#!/usr/bin/env python
# coding: utf-8
METAMAP INSTALL
https://metamap.nlm.nih.gov/Installation.shtml
https://github.com/AnthonyMRios/pymetamap
http://maraca.d.umn.edu/cgi-bin/umls_similarity/umls_similarity.cgi?word1=C0040866&word2=C0717386&sab=MSH&rel=PAR%2FCHD&similarity=path&sabdef=UMLS_ALL&reldef=CUI%2FPAR%2FCHD%2FRB%2FRN&relatedness=vector&button=Compute+Relatedness

bunzip2 -c public_mm_<platform>_<year>.tar.bz2 | tar xvf - 
cd public_mm
./bin/install.sh
	- <PATH TO public_mm>
	- <PATH TO (which java)>
./bin/skrmedpostctl start
./bin/wsdserverctl start
echo "lung cancer" | ./bin/metamap -I


from pymetamap import MetaMap
mm = MetaMap.get_instance('<path of metamap>')
sents = ['Heart Attack', 'John had a huge heart attack']
concepts,error = mm.extract_concepts(sents,[1,2])
for concept in concepts: 
	print (concept)


# In[1]:


import pandas as pd
import os
import json
import numpy as np
import time
from tqdm import tqdm


# In[2]:


import pymetamap
from pymetamap import MetaMap
mm = MetaMap.get_instance('<path of metamap>')


# In[3]:


print (os.getcwd())
os.chdir('homepath')
print (os.getcwd())


# In[4]:


file = 'srcdata'


# In[5]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')
import src.data.DataLoader as CustomDataLoader
from src.features.CustomTokenizer import CustomTokenizer


# In[6]:


pairs = CustomDataLoader.DataLoader(file)


# In[7]:


drugSemType = ['phsu','orch','vita','enzy','aapp','antb','chem','elii','enty','sbst','clnd'] #,'mnob'
freqSemType = ['tmco','qlco','qnco','inpr']
bodySemType = ['bsoj','blor','bpoc']
diseaseSemType = ['dsyn','inpo','mobd']


# In[8]:


# while True:
#     a = ['Patient arrives, via hospital wheelchair, Gait steady, History obtained from patient, Patient appears comfortable, Patient cooperative, alert, Oriented to person, place and time.']
#     b = ['Complex assessment performed, Patient arrives ambulatory, Gait steady, History obtained from, parent, Patient appears comfortable, Patient cooperative, alert, Oriented to person, place and time.']
#     #sents = [a, b]
#     drugnames, freqs, modes, diseases = [],[],[],[]
#     concepts,error = mm.extract_concepts(a,[1])
#     for concept in concepts: 
#         #print (type(concept))
#         #print (concept)
#         if isinstance(concept,pymetamap.Concept.ConceptMMI):
#             print (concept.index, concept.preferred_name, concept.cui, concept.semtypes,concept.semtypes[0])
#             drugnames += list(set([(concept.preferred_name, concept.cui) for each in drugSemType if each in concept.semtypes]))

#             freqs += list(set([(concept.preferred_name, concept.cui) for each in freqSemType if each in concept.semtypes]))
#             modes +=  list(set([(concept.preferred_name, concept.cui) for each in bodySemType if each in concept.semtypes]))
#             diseases +=  list(set([(concept.preferred_name, concept.cui) for each in diseaseSemType if each in concept.semtypes]))
#     print ("DRUGS:",drugnames)
#     print ("FREQUENCY:",freqs)
#     print ("MODE OF INTAKE:",modes)
#     print ("DISEASES:",diseases)

#     drugnames, freqs, modes, diseases = [],[],[],[]
#     concepts,error = mm.extract_concepts(b,[2])
#     for concept in concepts: 
#         #print (type(concept))
#         #print (concept)
#         if isinstance(concept,pymetamap.Concept.ConceptMMI):
#             print (concept.index, concept.preferred_name, concept.cui, concept.semtypes,concept.semtypes[0])
#             drugnames += list(set([(concept.preferred_name, concept.cui) for each in drugSemType if each in concept.semtypes]))

#             freqs += list(set([(concept.preferred_name, concept.cui) for each in freqSemType if each in concept.semtypes]))
#             modes +=  list(set([(concept.preferred_name, concept.cui) for each in bodySemType if each in concept.semtypes]))
#             diseases +=  list(set([(concept.preferred_name, concept.cui) for each in diseaseSemType if each in concept.semtypes]))
#     print ("DRUGS:",drugnames)
#     print ("FREQUENCY:",freqs)
#     print ("MODE OF INTAKE:",modes)
#     print ("DISEASES:",diseases)
#     input("WAIT")


# In[9]:


# def getConceptCui(a,b):
    
    
#     #mm1 = MetaMap.get_instance('/Users/kuntal/Desktop/ClinicalSTS/public_mm/bin/metamap16')
#     drugnamesA, freqsA, modesA, diseasesA, allA = [],[],[],[],[]
#     #print(a,b)
#     #input("B4")
#     concepts,error = mm.extract_concepts([a])
#     #print (concepts)
#     #input("IAMROOT")
#     for concept in concepts: 
#         #print (type(concept))
#         #print (concept)
#         if isinstance(concept,pymetamap.Concept.ConceptMMI):
# #             print (concept.index, concept.preferred_name, concept.cui, concept.semtypes,concept.semtypes[0])
#             #input("A")
#             allA.append(concept.cui)
# #             drugnamesA += list(set([(concept.preferred_name, concept.cui) for each in drugSemType if each in concept.semtypes]))
# #             freqsA += list(set([(concept.preferred_name, concept.cui) for each in freqSemType if each in concept.semtypes]))
# #             modesA +=  list(set([(concept.preferred_name, concept.cui) for each in bodySemType if each in concept.semtypes]))
# #             diseasesA +=  list(set([(concept.preferred_name, concept.cui) for each in diseaseSemType if each in concept.semtypes]))
#             ## This way bcoz it can match multiple patterns
#             drugnamesA += [concept.cui for each in drugSemType if each in concept.semtypes]
#             freqsA += [ concept.cui for each in freqSemType if each in concept.semtypes]
#             modesA +=  [ concept.cui for each in bodySemType if each in concept.semtypes]
#             diseasesA +=  [ concept.cui for each in diseaseSemType if each in concept.semtypes]
            
#     allA = list(set(allA))
#     drugnamesA = list(set(drugnamesA))
#     freqsA = list(set(freqsA))
#     modesA = list(set(modesA))
#     diseasesA = list(set(diseasesA))
# #     print ("DRUGS:",drugnamesA)
# #     print ("FREQUENCY:",freqsA)
# #     print ("MODE OF INTAKE:",modesA)
# #     print ("DISEASES:",diseasesA)
# #     print ("ALL:",allA)

#     drugnamesB, freqsB, modesB, diseasesB, allB = [],[],[],[],[]
#     concepts,error = mm.extract_concepts([b])
#     for concept in concepts: 
#         #print (type(concept))
#         #print (concept)
#         if isinstance(concept,pymetamap.Concept.ConceptMMI):
#             #print (concept.index, concept.preferred_name, concept.cui, concept.semtypes,concept.semtypes[0])
#             allB.append(concept.cui)
# #             drugnamesB += list(set([(concept.preferred_name, concept.cui) for each in drugSemType if each in concept.semtypes]))
# #             freqsB += list(set([(concept.preferred_name, concept.cui) for each in freqSemType if each in concept.semtypes]))
# #             modesB +=  list(set([(concept.preferred_name, concept.cui) for each in bodySemType if each in concept.semtypes]))
# #             diseasesB +=  list(set([(concept.preferred_name, concept.cui) for each in diseaseSemType if each in concept.semtypes]))
#             drugnamesB += [concept.cui for each in drugSemType if each in concept.semtypes]
#             freqsB += [ concept.cui for each in freqSemType if each in concept.semtypes]
#             modesB +=  [ concept.cui for each in bodySemType if each in concept.semtypes]
#             diseasesB +=  [ concept.cui for each in diseaseSemType if each in concept.semtypes]
            

#     allB = list(set(allB))
#     drugnamesB = list(set(drugnamesB))
#     freqsB = list(set(freqsB))
#     modesB = list(set(modesB))
#     diseasesB = list(set(diseasesB))
# #     print ("DRUGS:",list(set(drugnamesB)))
# #     print ("FREQUENCY:",freqsB)
# #     print ("MODE OF INTAKE:",modesB)
# #     print ("DISEASES:",diseasesB)
# #     print ("ALL:",allB)
    
#     ### Entity similarity
#     similarity  = len(set(allA).intersection(set(allB))) / (1.0* max(len(allA),len(allB)))
# #     print (similarity)
    
    
# #     input("WAIT:")
#     return (allA,allB),(drugnamesA,drugnamesB),(freqsA,freqsB),(modesA,modesB),(diseasesA,diseasesB),similarity


# In[10]:


len(pairs)


# In[11]:


pairs[0]


# In[13]:


conceptCui = []

alla,allb , druga, drugb, freqa,freqb, modea,modeb, disa,disb, sim = [],[],[],[],[],[],[],[],[],[],[]

for data in tqdm(pairs):
    sentA = data[0]
    sentB = data[1].strip("\n")
#     score = data[2]
    #print (sentA, sentB,data[2])
    

    #allAB, drugAB, freqAB, modeAB, disAB, similarity = getConceptCui(sentA, sentB)
    
    drugnamesA, freqsA, modesA, diseasesA, allA = [],[],[],[],[]
    #print(a,b)
    #input("B4")
    concepts,error = mm.extract_concepts([sentA])
    #print (concepts)
    #input("IAMROOT")
    for concept in concepts: 
        #print (type(concept))
        #print (concept)
        if isinstance(concept,pymetamap.Concept.ConceptMMI):
#             print (concept.index, concept.preferred_name, concept.cui, concept.semtypes,concept.semtypes[0])
            #input("A")
            allA.append(concept.cui)
#             drugnamesA += list(set([(concept.preferred_name, concept.cui) for each in drugSemType if each in concept.semtypes]))
#             freqsA += list(set([(concept.preferred_name, concept.cui) for each in freqSemType if each in concept.semtypes]))
#             modesA +=  list(set([(concept.preferred_name, concept.cui) for each in bodySemType if each in concept.semtypes]))
#             diseasesA +=  list(set([(concept.preferred_name, concept.cui) for each in diseaseSemType if each in concept.semtypes]))
            ## This way bcoz it can match multiple patterns
            drugnamesA += [concept.cui for each in drugSemType if each in concept.semtypes]
            freqsA += [ concept.cui for each in freqSemType if each in concept.semtypes]
            modesA +=  [ concept.cui for each in bodySemType if each in concept.semtypes]
            diseasesA +=  [ concept.cui for each in diseaseSemType if each in concept.semtypes]
            
#     allA = list(set(allA))
#     drugnamesA = list(set(drugnamesA))
#     freqsA = list(set(freqsA))
#     modesA = list(set(modesA))
#     diseasesA = list(set(diseasesA))
    allA = set(allA)
    drugnamesA = set(drugnamesA)
    freqsA = set(freqsA)
    modesA = set(modesA)
    diseasesA = set(diseasesA)
#     print ("DRUGS:",drugnamesA)
#     print ("FREQUENCY:",freqsA)
#     print ("MODE OF INTAKE:",modesA)
#     print ("DISEASES:",diseasesA)
#     print ("ALL:",allA)

    drugnamesB, freqsB, modesB, diseasesB, allB = [],[],[],[],[]
    concepts,error = mm.extract_concepts([sentB])
    for concept in concepts: 
        #print (type(concept))
        #print (concept)
        if isinstance(concept,pymetamap.Concept.ConceptMMI):
            #print (concept.index, concept.preferred_name, concept.cui, concept.semtypes,concept.semtypes[0])
            allB.append(concept.cui)
#             drugnamesB += list(set([(concept.preferred_name, concept.cui) for each in drugSemType if each in concept.semtypes]))
#             freqsB += list(set([(concept.preferred_name, concept.cui) for each in freqSemType if each in concept.semtypes]))
#             modesB +=  list(set([(concept.preferred_name, concept.cui) for each in bodySemType if each in concept.semtypes]))
#             diseasesB +=  list(set([(concept.preferred_name, concept.cui) for each in diseaseSemType if each in concept.semtypes]))
            drugnamesB += [concept.cui for each in drugSemType if each in concept.semtypes]
            freqsB += [ concept.cui for each in freqSemType if each in concept.semtypes]
            modesB +=  [ concept.cui for each in bodySemType if each in concept.semtypes]
            diseasesB +=  [ concept.cui for each in diseaseSemType if each in concept.semtypes]
            

    allB = set(allB)
    drugnamesB = set(drugnamesB)
    freqsB = set(freqsB)
    modesB = set(modesB)
    diseasesB = set(diseasesB)
#     print ("DRUGS:",list(set(drugnamesB)))
#     print ("FREQUENCY:",freqsB)
#     print ("MODE OF INTAKE:",modesB)
#     print ("DISEASES:",diseasesB)
#     print ("ALL:",allB)
    
    ### Entity similarity
    similarity  = len(allA.intersection(allB)) / (1.0* max(len(allA),len(allB)))
#     print (similarity)

    
    
#     alla.append(allAB[0])
#     allb.append(allAB[1])
#     druga.append(drugAB[0])
#     drugb.append(drugAB[1])
#     freqa.append(freqAB[0])
#     freqb.append(freqAB[1])
#     modea.append(modeAB[0])
#     modeb.append(modeAB[1])
#     disa.append(disAB[0])
#     disb.append(disAB[1])
#     sim.append(similarity)
    alla.append(allA)
    allb.append(allB)
    druga.append(drugnamesA)
    drugb.append(drugnamesB)
    freqa.append(freqsA)
    freqb.append(freqsB)
    modea.append(modesA)
    modeb.append(modesB)
    disa.append(diseasesA)
    disb.append(diseasesB)
    sim.append(similarity)
    


# In[14]:


print (len(alla),len(allb),len(druga),len(drugb),len(freqa),len(freqb),len(modea),len(modeb),len(disa),len(disb),len(sim))


# In[16]:


sentA = []
sentB = []
score = [0]*len(alla)
for data in tqdm(pairs):
    sentA.append(data[0])
    sentB.append(data[1].strip("\n"))
#     score.append(data[2].strip("\n"))


# In[17]:


cuidf = pd.DataFrame({'a':sentA,
              'b':sentB,
              'score':score,'alla':alla,'allb':allb,'druga':druga,'drugb':drugb,
                      'freqa':freqa,'freqb':freqb,'modea':modea,'modeb':modeb,
                      'disa':disa,'disb':disb,'cuisim':sim})


# In[18]:


cuidf['a_join'] = cuidf['a'].str.replace('"','').str.replace(" ","")
cuidf['b_join'] = cuidf['b'].str.replace('"','').str.replace(" ","")


# In[19]:


cuidf.head(20)


# In[20]:


cuidf.to_csv("cuidfTest.csv",index=False)


# In[ ]:




