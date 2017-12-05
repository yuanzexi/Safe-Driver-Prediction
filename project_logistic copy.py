
# In[116]:


miss_features = []
miss_perc = []
miss_cat = []
for f in train.columns:
    miss = train[train[f] == -1][f].count()
    if miss > 0:
        miss_features.append(f)
        if f.endswith('_cat'):
            miss_cat.append(f)
        miss_perc.append(miss*1.0/train.shape[0])


# In[117]:


for fea in miss_features:
    if train[fea].dtype == 'int64':
        train[fea].replace(-1,train[fea].mode().values[0],inplace=True)
    elif train[fea].dtype == 'float64':
        train[fea].replace(-1,np.nan,inplace=True)
        mean = train[fea].mean()
        train[fea].fillna(mean,inplace=True)


# In[118]:

scaler = StandardScaler()
train = scaler.fit_transform(train)
test = scaler.transform(test)
target = np.ravel(target)


lr = LogisticRegression(class_weight='balanced', C=0.1,n_jobs=-1,max_iter=200)

np.savetxt('LR_C_-3_1_8.txt', scores, delimiter=',')
saveFigure(c_arr,scores,'LR_C_-3_1_8')

