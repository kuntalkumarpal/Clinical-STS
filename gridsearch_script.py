import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
import math
import numpy as np
from hypopt import GridSearch
from sklearn.linear_model import LinearRegression
import xgboost

def selfun(x):
    return x


def personcorr(preds, dtrain):
    labels = dtrain.get_label()
    return 'error-pearson-corr',1-pearsonr(preds,labels)[0]


def cscorer(estimator,X_train, y_train):
    preds = estimator.predict(X_train)
    return pearsonr(preds,y_train)[0]

def most_common(lst):
    return max(set(lst), key=lst.count)

def avg(val):
    return (math.floor(val)+math.ceil(val))/2

class N2C2Classifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self , lr=1, ada=1, et=1, xgb=1, gb=1, rf=1, br=1, lasso=1, lrf=selfun , adaf=selfun, etf=selfun, xgbf=selfun, gbf=selfun, rff=selfun ,brf=selfun, lassof=selfun):
        """
        Called when initializing the classifier
        """
        self.lr = lr
        self.ada = ada
        self.et = et
        self.xgb = xgb
        self.gb = gb
        self.rf = rf
        self.br = br
        self.lasso = lasso
        self.lrf = lrf
        self.adaf = adaf
        self.etf = etf
        self.xgbf = xgbf
        self.gbf = gbf
        self.rff = rff
        self.lassof = lassof
        self.brf = brf

    def fit(self, X, y=None):
        return self
    
    def _weighted_scoring(self,x):
        if sum([self.lr,self.ada,self.et,self.xgb,self.gb,self.rf,self.br,self.lasso]) == 0:
            return x['lr']
        score1 = self.lr * self.lrf(x['lr'])
        score2 = self.ada * self.adaf(x['ada'])
        score3 = self.et * self.etf(x['et'])
        score4 = self.xgb * self.xgbf(x['xgb'])
        score5 = self.gb * self.gbf(x['gb'])
        score6 = self.rf * self.rff(x['rf'])
        score7 = self.br * self.brf(x['br'])
        score8 = self.lasso * self.lassof(x['lasso'])
        return sum([score1, score2, score3 ,score4, score5, score6, score7, score8] )/sum([self.lr, self.ada, self.et, self.xgb, self.gb, self.rf, self.br, self.lasso])
    
    def _max_scoring(self,x):
        if sum([self.lr,self.ada,self.et,self.xgb,self.gb,self.rf,self.br,self.lasso]) == 0:
            return x['et']
        score1 = self.lr * self.lrf(x['lr'])
        score2 = self.ada * self.adaf(x['ada'])
        score3 = self.et * self.etf(x['et'])
        score4 = self.xgb * self.xgbf(x['xgb'])
        score5 = self.gb * self.gbf(x['gb'])
        score6 = self.rf * self.rff(x['rf'])
        score7 = self.br * self.brff(x['br'])
        score8 = self.lasso * self.lassof(x['lasso'])
        scores = [score1,score2,score3,score4,score5,score6,score7,score8]
        scores = list(filter(lambda a: a != 0, scores))        \
        return most_common(scores)
    

    def predict(self, X, y=None):
        return([self._weighted_scoring(row) for index,row in X.iterrows()])

    def score(self, X, y=None):
        return(pearsonr(self.predict(X),y)[0]) 
    
df = pd.read_csv("multiRegressionOP.csv")
X_full = df[['lr','ada','et','xgb','gb','rf']]
y_full = df['gold']
X_train = X_full[0:1477]
y_train = y_full[0:1477]
X_dev = X_full[1477:]
y_dev = y_full[1477:]

print(X_full.head())
print("Length of Train/Test:",len(X_train),len(X_dev))

params = {
    'lr' : np.random.uniform(0,1,[5]).tolist() + [1.0],
    'ada' : np.random.uniform(0,1,[5]).tolist() + [1.0],
    'et' : np.random.uniform(0,1,[5]).tolist() + [1.0],
    'xgb' : np.random.uniform(0,1,5).tolist() + [1.0],
    'gb' : np.random.uniform(0,1,[5]).tolist() + [1.0],
    'rf' : np.random.uniform(0,1,[5]).tolist() + [1.0], 
    'br' : np.random.uniform(0,1,5).tolist() + [1.0], 
    'lasso' : np.random.uniform(0,1,[5]).tolist() + [1.0], 
    'lrf' : [math.floor,math.ceil,avg],
    'adaf' : [math.floor,math.ceil,avg],
    'etf' : [math.floor,math.ceil,avg],
    'xgbf' : [math.floor,math.ceil,avg],
    'gbf' : [math.floor,math.ceil,avg],
    'rff' :[math.floor,math.ceil,avg],
    'brf' : [math.floor,math.ceil,avg],
    'lassof' : [math.floor,math.ceil,avg],
}


model = GridSearch(N2C2Classifier(),param_grid=params)

model.fit(X_train,y_train,X_dev,y_dev,verbose=True)

print(model.get_best_params())
print(model.get_best_score())


