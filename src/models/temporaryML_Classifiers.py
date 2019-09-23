#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn
import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import StratifiedShuffleSplit 

os.chdir('homepath')

def getTrainDevTestSplit(X,Y):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    for trainIdx,otherIdx in  sss.split(X, Y):
        print("")
        
    train_pairs = list()
    train_labels = list()
    for idx in trainIdx:
        train_pairs.append(X[idx])
        train_labels.append(Y[idx])
        
    other_pairs = list()
    other_labels = list()
    for idx in otherIdx:
        other_pairs.append(X[idx])
        other_labels.append(Y[idx])
        
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3333, random_state=0)
    for devIdx,testIdx in  sss.split(other_pairs, other_labels):
        print("")
        
    dev_pairs = list()
    dev_labels = list()
    for idx in devIdx:
        dev_pairs.append(other_pairs[idx])
        dev_labels.append(other_labels[idx])
        
    test_pairs = list()
    test_labels = list()
    for idx in testIdx:
        test_pairs.append(other_pairs[idx])
        test_labels.append(other_labels[idx])
        
        
    XtrainData = np.array(train_pairs)
    YtrainData = np.array(train_labels)
    
    XvalData = np.array(dev_pairs)
    YvalData = np.array(dev_labels)
        
    XtestData = np.array(test_pairs)
    YtestData = np.array(test_labels)
    return XtrainData,YtrainData,XvalData,YvalData,XtestData,YtestData

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return int(np.floor(n*multiplier + 0.5) / multiplier)

def saveResultsAndPC_Score(YtestData,YpredtestData,outpath):
    testg = open(os.path.join(outpath,"testgs.txt"),'w')
    testp = open(os.path.join(outpath,"testsys.txt"),'w')
    
    for eachg, eachp in zip(YtestData.flatten().tolist(),YpredtestData.flatten().tolist()):
        testg.write(str(eachg)+"\n")
        testp.write(str(eachp)+"\n") 
    
    testg.close()
    testp.close()
    
    perlScriptPath = "data/train/raw/"
    os.path.join(perlScriptPath,"correlation-noconfidence.pl")
    os.system("perl "+os.path.join(perlScriptPath,"correlation-noconfidence.pl")+" "+ os.path.join(outpath,"testgs.txt")+" "+os.path.join(outpath,"testsys.txt") + ">"+os.path.join(outpath,"pc_score.txt"))


data = pd.read_csv("df.csv",delimiter=',')

data['Label'] = data[['Score']].apply(
    lambda row: round_half_up(row), axis=1)

Xfeatures = data[['Jaccard','Jaccard_G','Q2', 'Q3', 'Q4', 'Cosine', 'Dice', 'Overlap',
       'Tversky', 'MongeElkan','TfIdf','Affine','Bag','Jaro','Editex','Levenshtein','NeedlemanWunsch','SmithWaterman']]
Yfeatures = data[['Label']]

scaledData = Xfeatures.values
Y = Yfeatures.values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(scaledData)

trainSize = 0.7
valSize = 0.2
testSize = 0.1
dataSize = X.shape[0]


XtrainData,YtrainData,XvalData,YvalData,XtestData,YtestData = getTrainDevTestSplit(X,Y)

print (len(XtrainData), len(XvalData), len(XtestData))
print (len(YtrainData), len(YvalData), len(YtestData))

#"""
#    Logistic Regression
#"""
#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(XtrainData, YtrainData)
#
#YpredtestData=clf.predict(XtestData)
#
#outpath = "output/LogisticRegression"
#saveResultsAndPC_Score(YtestData,YpredtestData,outpath)
#
#"""
#    Decision Tree Classifier
#"""
#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(XtrainData, YtrainData)
#YpredtestData = clf.predict(XtestData)
#
#outpath = "output/DecisionTreeClassifier"
#saveResultsAndPC_Score(YtestData,YpredtestData,outpath)

"""
    XGBoost Classifier
"""
from models.gridSearchForXGBoost import TrainNTestXGBoost
YpredtestData = TrainNTestXGBoost(XtrainData, XtestData, YtrainData.flatten().tolist(), YtestData.flatten().tolist())

#outpath = "output/XGBoostClassifier"
#saveResultsAndPC_Score(YtestData,YpredtestData,outpath)
#
#"""
#    SVM One Vs One Classifier
#"""
#from sklearn import svm
#clf = svm.SVC(gamma='scale', decision_function_shape='ovo')
#clf = clf.fit(XtrainData, YtrainData)
#YpredtestData = clf.predict(XtestData)
#
#outpath = "output/SVM_OneVsOneClassifier"
#saveResultsAndPC_Score(YtestData,YpredtestData,outpath)
#
#"""
#    SVM One Vs All Classifier
#"""
#from sklearn import svm
#clf = svm.LinearSVC()
#clf = clf.fit(XtrainData, YtrainData)
#YpredtestData = clf.predict(XtestData)
#
#outpath = "output/SVM_OneVsAllClassifier"
#saveResultsAndPC_Score(YtestData,YpredtestData,outpath)