
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
# the only sample whose 'ps_car_12' value is -1 and the 'target' value is 0, I think this is a noise sample
train.drop(298018,axis=0,inplace=True) 

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

rf = ensemble.RandomForestClassifier(n_estimators = 94 , max_depth=5, 
                                 min_samples_leaf=4, n_jobs = -1,  
                                 max_features='auto', random_state=0, class_weight='balanced_subsample')
clf.fit(train,target)

test = pd.read_csv('test.csv')
test.drop(calc,axis=1,inplace=True)
ids = test['id'].values
# the only sample whose 'ps_car_12' value is -1 and the 'target' value is 0, I think this is a noise sample
test.drop('id',axis=1,inplace=True)
for feature in cat_features:
    dummies = pd.get_dummies(test[feature],prefix=feature,drop_first=True)
    test = pd.concat([test,dummies],axis=1)
    test.drop(feature,axis=1,inplace=True)

pre_proba = clf.predict_proba(test)
submit = pd.DataFrame({'id':ids,'target':pre_proba.T[1]})
submit.to_csv('xgb_submit.csv',index=False) 

