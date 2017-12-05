
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import seaborn as sns
import xgboost as xgb
#import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import random


# In[ ]:


train = pd.read_csv('train.csv')

calc = []
for i in train.columns[2:]:
    if i.startswith('ps_calc'):
        calc.append(i)
train.drop(calc,axis=1,inplace=True)
# found that 'ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin' are almost uniform distribution on target = 1
# the only sample whose 'ps_car_12' value is -1 and the 'target' value is 0, I think this is a noise sample
train.drop(298018,axis=0,inplace=True) 

# 'ps_reg_01','ps_reg_02','ps_reg_03' are correlated and their combination's distribution looks great, like a normal distribution
#train.drop(['ps_reg_01','ps_reg_02'],axis=1,inplace=True)
'''
# 'ps_car_12','ps_car_13' are correlated and their combination's distribution looks great
ps_car = train['ps_car_12'].add(train['ps_car_13'])#.add(train['ps_car_14'])
ps_car.name = 'ps_car_1213'
train = pd.concat([train,ps_car],axis=1)
#train.drop(['ps_car_12','ps_car_13','ps_car_14'],axis=1,inplace=True)
train.drop(['ps_car_12','ps_car_13'],axis=1,inplace=True)
'''
target = train['target']
train.drop('target',axis=1,inplace=True)
train.drop('id',axis=1,inplace=True)


# In[ ]:


cat_features = []
bin_features = []
continue_features = []
ordinal_features = []
for i in train.columns[2:]:
    if i.endswith('_bin'):
        bin_features.append(i)
    elif i.endswith('_cat'):
        cat_features.append(i)
    elif train[i].dtype == 'float64':
        continue_features.append(i)
    elif train[i].dtype == 'int64':
        ordinal_features.append(i)
        
for feature in cat_features:
    dummies = pd.get_dummies(train[feature],prefix=feature,drop_first=True)
    train = pd.concat([train,dummies],axis=1)
    train.drop(feature,axis=1,inplace=True)

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size = 0.25, random_state = 0)
'''
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''


# In[ ]:


def gini(y, pred):
    fpr, tpr, thr = metrics.roc_curve(y, pred, pos_label=1)
    g = 2 * metrics.auc(fpr, tpr) - 1
    return g

def gini_normalized(y,pred):
    return  gini(y, pred) / gini(y, y)
def cv_score(model,X,Y,cv=5):
    kf = StratifiedKFold(n_splits=cv)
    X = X.as_matrix()
    Y= Y.as_matrix()
    score = np.zeros(5)
    for train_index,test_index in kf.split(X,Y):
        train_x, test_x = X[train_index],X[test_index]
        train_y, test_y = Y[train_index],Y[test_index]
        model.fit(train_x,train_y)
        pre = model.predict(test_x)
        pre_proba = model.predict_proba(test_x).T[1]
        temp = [metrics.accuracy_score(pre,test_y),
                metrics.fbeta_score(test_y,pre,beta=2.0),
		metrics.average_precision_score(test_y,pre_proba),
                metrics.roc_auc_score(test_y,pre_proba),
                gini_normalized(test_y,pre_proba)]
        score = np.mean([score,temp],axis=0)
    return score
def saveFigure(x,scores,x_label):
    scores = np.matrix(scores)
    fig = plt.figure(figsize = (12,10))
    length = len(np.ravel(scores[0]))
    data = pd.DataFrame(data=scores,columns=['acc','F2 score','ROC_AUC','Gini'],index=x)
    plt.plot(data)
    plt.xlabel(x_label,fontsize=18)
    plt.ylabel('CV-scores',fontsize=18)
    plt.legend(['acc','F2 score','ROC_AUC','Gini'])
    #plt.show()
    fig.savefig(x_label+'.png', dpi=fig.dpi)


# In[ ]:


weights = [0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.5]
weight_scores = []
for weight in weights:
    clf = xgb.XGBClassifier(n_estimators = 90, max_depth = 5, silent = True, n_jobs = -1,gamma=2,
                        booster='gbtree',random_state=7, subsample = 0.8, colsample_bytree = 0.8,
                        learning_rate=0.1, objective = 'binary:logistic',scale_pos_weight=weight)#scale_pos_weight
    weight_scores.append(cv_score(clf,x_train,y_train))
weight_scores = np.array(weight_scores)

np.savetxt('xgb_weights_1_30_0.1_90_5.txt', weight_scores, delimiter=',')
#saveFigure(weights, weight_scores, 'xgb_weight_1_2_0.1_90_5')

