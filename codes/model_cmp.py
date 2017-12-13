

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
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
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
train.drop(298018,axis=0,inplace=True) 

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
def process_missing(data):

    miss_features = []
    miss_perc = []
    miss_cat = []
    for f in data.columns:
        miss = data[data[f] == -1][f].count()
        if miss > 0:
            miss_features.append(f)
            if f.endswith('_cat'):
                miss_cat.append(f)
            miss_perc.append(miss*1.0/data.shape[0])


    # In[117]:


    for fea in miss_features:
        if data[fea].dtype == 'int64':
            data[fea].replace(-1,data[fea].mode().values[0],inplace=True)
        elif data[fea].dtype == 'float64':
            data[fea].replace(-1,np.nan,inplace=True)
            mean = data[fea].mean()
            data[fea].fillna(mean,inplace=True)

    return data

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
    data = pd.DataFrame(data=scores,columns=['acc','F2 score','AP','ROC_AUC','Gini'],index=x)
    plt.plot(data)
    plt.xlabel(x_label,fontsize=18)
    plt.ylabel('CV-scores',fontsize=18)
    plt.legend(['acc','F2 score','AP','ROC_AUC','Gini'])
    #plt.show()
    fig.savefig(x_label+'.png', dpi=fig.dpi)


# In[ ]:



# In[19]:


def test_score(model,test_x,test_y):
    #model.fit(X,Y)
    pre = model.predict(test_x)
    pre_proba = model.predict_proba(test_x).T[1]
    score = [metrics.accuracy_score(pre,test_y),
            metrics.fbeta_score(test_y,pre,beta=2.0),
            metrics.average_precision_score(test_y,pre_proba),
            metrics.roc_auc_score(test_y,pre_proba),
            gini_normalized(test_y,pre_proba)]
    return score


# In[ ]:


clf_xgb = xgb.XGBClassifier(n_estimators = 90, max_depth = 5, silent = True, n_jobs = -1,
                    seed=7, subsample = 0.8, colsample_bytree = 0.8,gamma=2,
                    learning_rate=0.1, objective = 'binary:logistic')#scale_pos_weight

clf_rf = ensemble.RandomForestClassifier(n_estimators= 120, max_depth=9, 
                                         min_samples_leaf=7, n_jobs = -1, #oob_score =True, 
                                         max_features='auto', random_state=0, class_weight='balanced_subsample')

clf_lr = LogisticRegression(class_weight='balanced', C=0.1,n_jobs=-1,max_iter=200)


# In[ ]:


clf_xgb.fit(x_train,y_train)
xgb_final = (test_score(clf_xgb,x_test,y_test))
clf_rf.fit(x_train,y_train)
rf_final = (test_score(clf_rf,x_test,y_test))
x_train = process_missing(x_train)
clf_lr.fit(x_train,y_train)
x_test = process_missing(x_test)
lr_final = (test_score(clf_lr,x_test,y_test))


# In[ ]:


'''
kf = StratifiedKFold(n_splits=5,random_state=7)
xgb_score = []
rf_score = []
lr_score = []

train2= train.as_matrix()
target2= target.as_matrix()
for train_index,test_index in kf.split(train,target):
    train_x, test_x = train2[train_index],train2[test_index]
    train_y, test_y = target2[train_index],target2[test_index]
    clf_xgb.fit(train_x,train_y)
    xgb_score.append(test_score(clf_xgb,test_x,test_y))
    clf_rf.fit(train_x,train_y)
    rf_score.append(test_score(clf_rf,test_x,test_y))

process_missing(train)
train = train.as_matrix()
target = target.as_matrix()
for train_index,test_index in kf.split(train,target):
    train_x, test_x = train[train_index],train[test_index]
    train_y, test_y = target[train_index],target[test_index]
    clf_lr.fit(train_x,train_y)
    lr_score.append(test_score(clf_lr,test_x,test_y))


# In[ ]:

print ('xgb:',xgb_score)
print ('rf:',rf_score)
print ('lr:',lr_score)


xgb_final = np.mean(xgb_score,axis=0)
rf_final = np.mean(rf_score,axis=0)
lr_final = np.mean(lr_score,axis=0)
# In[ ]:
'''
final_score = [xgb_final,rf_final,lr_final]
print ('final score',final_score)

np.savetxt('final_score.txt', final_score, delimiter=',')


# In[ ]:


#pd.DataFrame( data= final_score,
#            columns=['acc','F2 score','AP','ROC_AUC','Gini'],index=['xgb','rf','lr'])

