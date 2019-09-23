#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
 
import numpy as np
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV 
sys.path.append('xgboost/wrapper/')
import xgboost as xgb
 
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import f1_score,accuracy_score,roc_curve,auc,balanced_accuracy_score
from joblib import load
class XGBoostClassifier():
    def __init__(self, num_boost_round=10, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'multi:softprob'})
 
    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        print(dtrain.num_row())
        self.clf = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=num_boost_round)
 
    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])
 
    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)
 
    def score(self, X, y):
        Y = self.predict_proba(X)
        return 1 / logloss(y, Y)
 
    def get_params(self, deep=True):
        return self.params
 
    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self
    
    def printDifferentMetricScores(self, X_Test,y_test):
        dtest = xgb.DMatrix(X_Test)
        y_pred_LR = self.clf.predict(dtest)
        y_pred_prob_LR = self.clf.predict_proba(dtest)[:,1]
        
        f1_LR = f1_score(y_test,y_pred_LR,average='macro')
        accuracy_LR = accuracy_score(y_test,y_pred_LR)
        balanced_accuracy_LR = balanced_accuracy_score(y_test,y_pred_LR)
        fpr_LR,tpr_LR,threshods_LR = roc_curve(y_test, y_pred_prob_LR , pos_label=1)
        auc_LR = auc(fpr_LR,tpr_LR)
        
        print("F1 Score: {0}".format(f1_LR ))
        print("Accuracy Score: {0}".format(accuracy_LR))
        print("Balanced Accuracy: {0}".format(balanced_accuracy_LR))
        print("Area Under the ROC Curve: {0}".format(auc_LR))
        
        return y_pred_LR
        
    
def logloss(y_true, Y_pred):
    print("IN LOG LOSS \n")
    label2num = dict((name, i) for i, name in enumerate(sorted(set(y_true))))
    return -1 * sum(math.log(y[label2num[label]]) if y[label2num[label]] > 0 else -np.inf for y, label in zip(Y_pred, y_true)) / len(Y_pred)


def TrainNTestXGBoost(X_train_transformed,X_test_transformed,y_train,y_test):
    clf = XGBoostClassifier(
        eval_metric = 'auc',
        num_class = 6,
        nthread = 4,
        silent = 1,
        )
    parameters = {
        'num_boost_round': [100, 250, 500],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
    }
    clf = GridSearchCV(clf, parameters, n_jobs=1, cv=2)
    
    clf.fit(X_train_transformed, y_train)
    print(clf.cv_results_)
#    score = max(clf.cv_results_, key=lambda x: x[1])
#    print('score:', score)
#    for param_name in sorted(best_parameters.keys()):
#        print("%s: %r" % (param_name, best_parameters[param_name]))
    #print('predicted:', clf.predict([[1,1]]))
    return clf.predict(X_test_transformed)


#if __name__ == '__main__':
#    YpredtestData1 = TrainNTestXGBoost(XtrainData, XtestData, YtrainData.flatten().tolist(), YtestData.flatten().tolist())
