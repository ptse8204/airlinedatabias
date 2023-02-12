#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

#!git clone https://github.com/Trusted-AI/AIF360.git
#!pip install aif360[all]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset
from aif360.explainers import MetricTextExplainer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric

from collections import defaultdict

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow.compat.v1 as tf

#ticket=pd.read_csv("Origin_and_Destination_Survey_DB1BTicket_2022_1.csv",nrows=1000000)
#income=pd.read_csv("median_income_cityID.csv")
#race=pd.read_csv("race_id.csv")
#market=pd.read_csv("Origin_and_Destination_Survey_DB1BMarket_2022_1.csv",nrows=1000000)

def bias_mitigation(ticket,market,race,income):
    #input datasets
    #outputs bias statistics and returns aif train,val and test dataframes 
    #initial data and merges
    ticket=ticket[["ItinID", "Coupons", 'Year', 'Quarter','Origin', 'OriginCityMarketID', 'OriginState',
                'RoundTrip', 'OnLine', 'DollarCred', 'FarePerMile','RPCarrier', 'Passengers', 'ItinFare', 'BulkFare',
               'MilesFlown', 'ItinGeoType']]
    
    #not including roundtrips, comment this out to include them
    #ticket=ticket[ticket["RoundTrip"]!=1]



    market=market[["ItinID", 'Year', 'Quarter','Origin', 'DestCityMarketID', 'OriginState',
                    'RPCarrier', 'Passengers', 'BulkFare','ItinGeoType']]



    ticket=ticket.merge(race,left_on="OriginCityMarketID",right_on="Code",how="left",suffixes=('', '_origin'))
    ticket=ticket.merge(market,on="ItinID",how="left",suffixes=('', '_y'))
    ticket=ticket.merge(race,left_on="DestCityMarketID",right_on="Code",how="left",suffixes=('', '_dest'))
    ticket=ticket.merge(income,left_on="Code",right_on="Code",how="left",suffixes=('', '_origin'))

    #creating a aif360dataframe and new columns
    def aif360_dataframe(df):
        def flight_length(x):
            if x <= 1725: #miles
                return "short-haul"
            elif x>1725 and x<=3450:
                return "medium-haul"
            else:
                return "long-haul"

        def majority(x,y):
            #x=white,y=non-white
            #This is used to find local majority of population
            #and return a 1 (white) or 0 (non-white)
            if x >= y:
                return 1.0
            elif y>x:
                return 0.0
            else:
                return 0.0
        def income_median(x):
            #x=white,y=non-white
            #This is used to classify local income median of population
            #and return a 1 (high) or 0 (low)
            if x > income_quant:
                return 1.0
            elif x<=income_quant:
                return 0.0
            else:
                return 0.0
        def fare_classes(x):
            #the output we are trying to predict
            # 1 is the preferred result 
            # 0 is everything else
            # we should discuss how we want to do this
            if x < fpm_quant: #if x is less than the 75 quantile, we get a favorable result
                return 1.0
            elif x>=fpm_quant: #we get unfavorable result
                return 0.0
            else:
                return 0.0
        #creating new columns

        df["flight_length"]=df["MilesFlown"].apply(lambda x: flight_length(x))
        #df=df[df["flight_length"]=="short-haul"]
        df=df.dropna(subset=["median_income"])

        fpm_quant=np.quantile(df["FarePerMile"],0.75) #third quantile
        fare_class_series=df["FarePerMile"].apply(lambda x: fare_classes(x)) 
        df=df.assign(fare_class=fare_class_series)

        df["local_majority_origin"]=df.apply(lambda x: majority(x["White alone proportion"],x["Non-White alone proportion"]),axis=1)
        df["local_income_origin"]=df.apply(lambda x: income_median(x["median_income"]),axis=1)
        df["local_majority_dest"]=df.apply(lambda x: majority(x["White alone proportion_dest"],x["Non-White alone proportion_dest"]),axis=1)
        new_df=df.copy()
        #aif360 parameters
        label_name='fare_class'
        favorable_classes=[1.0]
        #using origin
        protected_attribute_names=['local_income_origin']
        privileged_classes=[[1.0]]
        #example features,add or change features here,this is using origin
        features_keep=["local_income_origin","MilesFlown","local_majority_origin","local_majority_dest"] 
        #aif dataframe
        aif_data=StandardDataset(df,label_name,favorable_classes,protected_attribute_names,privileged_classes,features_to_keep=features_keep
                   ,metadata={'protected_attribute_maps': [{1.0: 'high-income', 0.0: 'low-income'}]})
        #more aif parameters
        sens_ind = 0
        sens_attr = aif_data.protected_attribute_names[sens_ind]

        unprivileged_groups = [{sens_attr: v} for v in
                               aif_data.unprivileged_protected_attributes[sens_ind]]
        privileged_groups = [{sens_attr: v} for v in
                             aif_data.privileged_protected_attributes[sens_ind]]
        (aif_data_train,aif_data_val,aif_data_test) = aif_data.split([0.5, 0.8], shuffle=True)
        
        #this is for metrics from the training set
        metric_fpm = BinaryLabelDatasetMetric(
            aif_data_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)

        metric_fpm_explainer = MetricTextExplainer(metric_fpm)

        print(metric_fpm_explainer.disparate_impact(),"\n")
        return new_df,aif_data_train,aif_data_val,aif_data_test,unprivileged_groups,privileged_groups,sens_attr



    #income threshold
    income_quant=income["median_income"].quantile(0.25)
    #groupbys
    ticket=ticket.groupby(["OriginCityMarketID","DestCityMarketID","RPCarrier","Coupons"]).median().reset_index()
    
    new_df,aif_data_train,aif_data_val,aif_data_test,unprivileged_groups,privileged_groups,sens_attr=aif360_dataframe(ticket)




    def describe(train=None, val=None, test=None):
        if train is not None:
            display(Markdown("#### Training Dataset shape"))
            print(train.features.shape)
        if val is not None:
            display(Markdown("#### Validation Dataset shape"))
            print(val.features.shape)
        display(Markdown("#### Test Dataset shape"))
        print(test.features.shape)
        display(Markdown("#### Favorable and unfavorable labels"))
        print(test.favorable_label, test.unfavorable_label)
        display(Markdown("#### Protected attribute names"))
        print(test.protected_attribute_names)
        display(Markdown("#### Privileged and unprivileged protected attribute values"))
        print(test.privileged_protected_attributes, 
              test.unprivileged_protected_attributes)
        display(Markdown("#### Dataset feature names"))
        print(test.feature_names)

    # return information on train,val,and test
    describe(aif_data_train,aif_data_val,aif_data_test)



    dataset = aif_data_train
    model = make_pipeline(StandardScaler(),LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

    lr_simple = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)



    def test(dataset, model, thresh_arr):
        try:
            # sklearn classifier
            y_val_pred_prob = model.predict_proba(dataset.features)
            pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
        except AttributeError:
            # aif360 inprocessing algorithm
            y_val_pred_prob = model.predict(dataset).scores
            pos_ind = 0

        metric_arrs = defaultdict(list)
        for thresh in thresh_arr:
            y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

            dataset_pred = dataset.copy()
            dataset_pred.labels = y_val_pred
            metric = ClassificationMetric(
                    dataset, dataset_pred,
                    unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)

            metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                         + metric.true_negative_rate()) / 2)
            metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
            metric_arrs['disp_imp'].append(metric.disparate_impact())
            metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
            metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
            metric_arrs['theil_ind'].append(metric.theil_index())

        return metric_arrs


    # # Logistic Regression
    print("Simple Logistic Regression Validation:\n")

    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_data_val,
                       model=lr_simple,
                       thresh_arr=thresh_arr)
    lr_simple_best_ind = np.argmax(val_metrics['bal_acc'])

    def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.plot(x, y_left)
        ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
        ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
        ax1.xaxis.set_tick_params(labelsize=14)
        ax1.yaxis.set_tick_params(labelsize=14)
        ax1.set_ylim(0.3, 0.8)

        ax2 = ax1.twinx()
        ax2.plot(x, y_right, color='r')
        ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
        if 'DI' in y_right_name:
            ax2.set_ylim(0., 0.7)
        else:
            ax2.set_ylim(-0.25, 0.1)

        best_ind = np.argmax(y_left)
        ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
        ax2.yaxis.set_tick_params(labelsize=14)
        ax2.grid(True)


    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')

    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.')


    def describe_metrics(metrics, thresh_arr):
        best_ind = np.argmax(metrics['bal_acc'])
        print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
        print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
        #disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
        disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
        print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
        print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
        print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
        print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
        print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
        print("\n")




    describe_metrics(val_metrics, thresh_arr)


    print("Simple Logistic Regression Test:\n")


    lr_simple_metrics = test(dataset=aif_data_test,
                           model=lr_simple,
                           thresh_arr=[thresh_arr[lr_simple_best_ind]])




    describe_metrics(lr_simple_metrics, [thresh_arr[lr_simple_best_ind]])


    # # RandomForest
    
    print("Simple RandomForest Validation:\n")



    dataset = aif_data_train
    model = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=100, min_samples_leaf=25,random_state=1))
    fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
    rf_simple= model.fit(dataset.features, dataset.labels.ravel(), **fit_params)



    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_data_val,
                       model=rf_simple,
                       thresh_arr=thresh_arr)
    rf_simple_best_ind = np.argmax(val_metrics['bal_acc'])


    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')



    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.')



    describe_metrics(val_metrics, thresh_arr)

    print("Simple RandomForest Test:\n")
    rf_simple_metrics = test(dataset=aif_data_test,
                           model=rf_simple,
                           thresh_arr=[thresh_arr[rf_simple_best_ind]])


    describe_metrics(rf_simple_metrics, [thresh_arr[rf_simple_best_ind]])


    # # Bias Mitigation
    print("Bias Mitigation")

    # ## Preprocessing
    print("Preprocessing")

    # ### Disparate Impact Remover
    print("Disparate Impact Remover")



    DI = DisparateImpactRemover(repair_level=1.0,sensitive_attribute=sens_attr)
    dataset_transf_train = DI.fit_transform(aif_data_train)



    metric_transf_train = BinaryLabelDatasetMetric(
            dataset_transf_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
    explainer_transf_train = MetricTextExplainer(metric_transf_train)

    print(explainer_transf_train.disparate_impact())


    # #### Logistic Regression
    print("LR Disparate Impact Remover Validation:\n")


    dataset = dataset_transf_train
    model = make_pipeline(StandardScaler(),LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
    lr_transf = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)




    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_data_val,
                       model=lr_transf,
                       thresh_arr=thresh_arr)
    lr_transf_best_ind = np.argmax(val_metrics['bal_acc'])




    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')



    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.')


    describe_metrics(val_metrics, thresh_arr)


    print("LR Disparate Impact Remover Test:\n")

    lr_transf_metrics = test(dataset=aif_data_test,
                             model=lr_transf,
                             thresh_arr=[thresh_arr[lr_transf_best_ind]])





    describe_metrics(lr_transf_metrics, [thresh_arr[lr_transf_best_ind]])


    # #### RandomForest
    print("RF Disparate Impact Remover Validation:\n")


    dataset = dataset_transf_train
    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=100, min_samples_leaf=25,random_state=1))
    fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
    rf_transf = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)



    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_data_val,
                       model=rf_transf,
                       thresh_arr=thresh_arr)
    rf_transf_best_ind = np.argmax(val_metrics['bal_acc'])



    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')



    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.')


    describe_metrics(val_metrics, thresh_arr)


    print("RF Disparate Impact Remover Test:\n")


    rf_transf_metrics = test(dataset=aif_data_test,
                             model=rf_transf,
                             thresh_arr=[thresh_arr[rf_transf_best_ind]])



    describe_metrics(rf_transf_metrics, [thresh_arr[rf_transf_best_ind]])


    # ### Reweighing
    print("Reweighing")

    
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_train = RW.fit_transform(aif_data_train)



    metric_transf_train = BinaryLabelDatasetMetric(
            dataset_transf_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
    explainer_transf_train = MetricTextExplainer(metric_transf_train)

    print(explainer_transf_train.disparate_impact())


    # #### Logistic Regression
    

    print("LR Reweighing Validation:\n")


    dataset = dataset_transf_train
    model = make_pipeline(StandardScaler(),LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': dataset.instance_weights}
    lr_transf = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)


    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_data_val,
                       model=lr_transf,
                       thresh_arr=thresh_arr)
    lr_transf_best_ind = np.argmax(val_metrics['bal_acc'])



    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')


    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.')



    describe_metrics(val_metrics, thresh_arr)


    print("LR Reweighing Test:\n")


    lr_transf_metrics = test(dataset=aif_data_test,
                             model=lr_transf,
                             thresh_arr=[thresh_arr[lr_transf_best_ind]])


    describe_metrics(lr_transf_metrics, [thresh_arr[lr_transf_best_ind]])


    # #### RandomForest
    

    print("RF Reweighing Validation:\n")


    dataset = dataset_transf_train
    model = make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=100, min_samples_leaf=25,random_state=1))
    fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}
    rf_transf = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)




    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_data_val,
                       model=rf_transf,
                       thresh_arr=thresh_arr)
    rf_transf_best_ind = np.argmax(val_metrics['bal_acc'])


    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')



    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.')


    describe_metrics(val_metrics, thresh_arr)

    print("RF Reweighing Test:")

    rf_transf_metrics = test(dataset=aif_data_test,
                             model=rf_transf,
                             thresh_arr=[thresh_arr[rf_transf_best_ind]])


    describe_metrics(rf_transf_metrics, [thresh_arr[rf_transf_best_ind]])


    # ## Inprocessing
    print("Inprocessing")

    # ### Prejudice Remover
    print("Prejudice Remover")


    model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
    pr_scaler = StandardScaler()

    dataset = aif_data_train.copy()
    dataset.features = pr_scaler.fit_transform(dataset.features)

    PR = model.fit(dataset)


    print("Prejudice Remover Validation:\n")


    thresh_arr = np.linspace(0.01, 0.50, 50)

    dataset = aif_data_val.copy()
    dataset.features = pr_scaler.transform(dataset.features)

    val_metrics = test(dataset=dataset,
                       model=PR,
                       thresh_arr=thresh_arr)
    pr_best_ind = np.argmax(val_metrics['bal_acc'])



    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')


    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.')


    describe_metrics(val_metrics, thresh_arr)


    print("Prejudice Remover Test:\n")



    dataset = aif_data_test.copy()
    dataset.features = pr_scaler.transform(dataset.features)

    pr_metrics = test(dataset=dataset,
                           model=PR,
                           thresh_arr=[thresh_arr[pr_best_ind]])


    describe_metrics(pr_metrics, [thresh_arr[pr_best_ind]])


    # ### AdversarialDebiasing

    print("Adversarial Debiasing")

    tf.disable_eager_execution()
    sess = tf.Session()
    dataset = aif_data_train.copy()
    ad_scaler = StandardScaler()
    AD=AdversarialDebiasing(privileged_groups = privileged_groups,unprivileged_groups = unprivileged_groups,
                            scope_name='debiased',debias=True,sess=sess)

    dataset.features = ad_scaler.fit_transform(dataset.features)

    AD = AD.fit(dataset)


    print("Adversarial Debiasing Validation:\n")


    thresh_arr = np.linspace(0.01, 0.50, 50)

    dataset = aif_data_val.copy()
    dataset.features = ad_scaler.transform(dataset.features)

    val_metrics = test(dataset=dataset,
                       model=AD,
                       thresh_arr=thresh_arr)
    ad_best_ind = np.argmax(val_metrics['bal_acc'])




    disp_imp = np.array(val_metrics['disp_imp'])
    disp_imp_err = 1 - np.minimum(disp_imp, 1/disp_imp)
    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         disp_imp_err, '1 - min(DI, 1/DI)')



    plot(thresh_arr, 'Classification Thresholds',
         val_metrics['bal_acc'], 'Balanced Accuracy',
         val_metrics['avg_odds_diff'], 'avg. odds diff.')




    describe_metrics(val_metrics, thresh_arr)


   
    print("Adversarial Debiasing Test:\n")


    dataset = aif_data_test.copy()
    dataset.features =ad_scaler.transform(dataset.features)

    ad_metrics = test(dataset=dataset,
                           model=AD,
                           thresh_arr=[thresh_arr[ad_best_ind]])


    describe_metrics(ad_metrics, [thresh_arr[ad_best_ind]])
    
    return aif_data_train,aif_data_val,aif_data_test



#bias_mitigation(ticket,market,race,income)

