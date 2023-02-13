#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#initial DecisionTreeClassifier model 
def init_dec_tree(data):
    model =  DecisionTreeClassifier(class_weight = "balanced") 
    
    model.fit(data[["Distance", "Coupons"]], data['FareClass'])
    
    model.predict(data[["Distance", "Coupons"]])
    
    #calculate accuracy 
    
    score = model.score(data[["Distance", "Coupons"]], data['FareClass'])
    
    return score


# In[ ]:


def init_ran_forest(data):
    #create train test split
    X = df[['Coupons', 'OriginWac', 'Distance', 'DistanceGroup', 'Metro Area','White alone proportion', 
            'Non-White alone proportion']] 
    Y = df['FareClass']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, train_size = 0.75, random_state = 42)
    
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train,y_train) 
    return classifier.score( X_test, y_test)


# In[ ]:


def random_forest_aif360(df):
    #takes in a aif360 dataframe 
    model = make_pipeline(StandardScaler(),
                      RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
    fit_params = {'randomforestclassifier__sample_weight': training_aif.instance_weights}
    rf_orig_panel19 = model.fit(training_aif.features,training_aif.labels.ravel(), **fit_params)
    


# In[ ]:


def reweighing_trans(df):
    sens_ind = 0
    sens_attr = df.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                       df.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                     df.privileged_protected_attributes[sens_ind]]
    
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
    dataset_transf = RW.fit_transform(pre_df)
    return dataset_transf

