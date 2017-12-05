rf = ensemble.RandomForestClassifier(n_estimators=N_forest , max_depth=5, 
                                 min_samples_leaf=4, n_jobs = -1,  
                                 max_features='auto', random_state=0, class_weight='balanced_subsample')
rf.fit(train, target)
pre_proba = rf.predict_proba(test)

submit = pd.DataFrame({'id':test['id'],'target':pre_proba})
submit.to_csv('../input/submit.csv',index=False) 
