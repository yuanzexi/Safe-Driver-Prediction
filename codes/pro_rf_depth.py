
# coding: utf-8

# In[19]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn import *
#import xgboost as xgb
#import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cross_validation import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import StratifiedKFold
import random
train = pd.read_csv('train.csv')


# In[21]:

# found that 'ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin' are almost uniform distribution on target = 1
train.drop(['ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin'],axis=1,inplace=True)
# found that 'ps_calc_01','ps_calc_02','ps_calc_03' are uniform distribution on target = 1
train.drop(['ps_calc_01','ps_calc_02','ps_calc_03'],axis=1,inplace=True)
# found that 'ps_calc_04','ps_calc_09' are uniform distribution on target = 1
train.drop(['ps_calc_04','ps_calc_09'],axis=1,inplace=True)
# the only sample whose 'ps_car_12' value is -1 and the 'target' value is 0, I think this is a noise sample
train.drop(298018,axis=0,inplace=True) 

# 'ps_reg_01','ps_reg_02','ps_reg_03' are correlated and their combination's distribution looks great, like a normal distribution
'''
ps_reg = train['ps_reg_01'].add(train['ps_reg_02']).add(train['ps_reg_03'])
ps_reg.name = 'ps_reg'
train = pd.concat([train,ps_reg],axis=1)
train.drop(['ps_reg_01','ps_reg_02','ps_reg_03'],axis=1,inplace=True)
#train.drop(['ps_reg_01','ps_reg_02'],axis=1,inplace=True)

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



# In[22]:

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
    dummies = pd.get_dummies(train[feature],prefix=feature, drop_first=True)
    train = pd.concat([train,dummies],axis=1)
    train.drop(feature,axis=1,inplace=True)

x_train, x_test, y_train, y_test = train_test_split(train, target, test_size = 0.25, random_state = 0)
'''
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
'''


# In[42]:

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
                average_precision_score(test_y,pre_proba),
                metrics.roc_auc_score(test_y,pre_proba),
                gini_normalized(test_y,pre_proba)]
        score = np.mean([score,temp],axis=0)
    return score
def saveFigure(x,scores,x_label):
    scores = np.matrix(scores)
    fig = plt.figure(figsize = (12,10))
    length = len(np.ravel(scores[0]))
    data = pd.DataFrame(data=scores,columns=['acc','F2 score','AP','ROC_AUC','Gini'],index=x)
    plt.plot(data)
    plt.xlabel(x_label,fontsize=18)
    plt.ylabel('CV-scores',fontsize=18)
    plt.legend(['acc','F2 score','AP','ROC_AUC','Gini'])
    #plt.show()
    fig.savefig(x_label+'.png', dpi=fig.dpi)

'''
# In[38]:

N_forest = 200
#'classifier__n_estimators': [20, 30, 50],
                    #'classifier__max_depth': [2, 4],
                    #'classifier__min_samples_leaf': [2, 4]
rf = ensemble.RandomForestClassifier(n_estimators=N_forest , max_depth=5, 
                                 min_samples_leaf=4, n_jobs = -1, oob_score =True, 
                                 max_features='auto', random_state=0, class_weight='balanced_subsample')
rf.fit(x_train, y_train)
#features = train.columns.values
print("----- Training Done -----")


# In[39]:

votes = rf.estimators_
vote_results = []
for vote in votes:
    vote_results.append(vote.predict_proba(x_test).T[1])
mat = np.array(vote_results)


# In[40]:

tree_scores = []
#res = np.mean(mat[:10],axis=0)
for i in range(10,N_forest):
    pre_proba = np.mean(mat[:i],axis=0)
    pre = pre_proba[:]
    pre[pre >= 0.5] = 1
    pre[pre < 0.5] = 0
    temp = [metrics.accuracy_score(pre,y_test),
                metrics.fbeta_score(y_test,pre,beta=2.0),
                average_precision_score(y_test,pre_proba),
                metrics.roc_auc_score(y_test,pre_proba),
                gini_normalized(y_test,pre_proba)]
    tree_scores.append(temp)


# In[41]:

saveFigure(range(10,300),tree_scores,'RF_estimators_ap')
np.savetxt('rf_tree_10_300_ap.txt', tree_scores, delimiter=',')
#print(tree_scores)
'''

# In[34]:

depths = np.linspace(1,20,20,dtype=int)
depth_scores = []
for depth in depths:
    rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=depth, 
                                         min_samples_leaf=4, n_jobs = -1, oob_score =True, 
                                         max_features='auto', random_state=0, class_weight='balanced_subsample')
    depth_scores.append(cv_score(rf,x_train,y_train))


# In[35]:

saveFigure(depths,depth_scores,'RF_depths_1_20')
np.savetxt('rf_depth_1_20.txt', depth_scores, delimiter=',')
#print(scores)
'''

# In[36]:

#   leaves have closed relation with depth. 
#  when depth is small, large size of leaves makes no sense
leafs = np.linspace(1,10,10,dtype=int)
leaf_scores = []
for leaf in leafs:
    rf = ensemble.RandomForestClassifier(n_estimators=100, max_depth=5, 
                                         min_samples_leaf = int(leaf), n_jobs = -1, oob_score =True, 
                                         max_features='auto', random_state=0, class_weight='balanced_subsample')
    leaf_scores.append(cv_score(rf,x_train,y_train))
    


# In[ ]:

saveFigure(leafs,leaf_scores,'RF_leafs')
np.savetxt('rf_leaf_1_20.txt', leaf_scores, delimiter=',')
#print(scores)
'''
