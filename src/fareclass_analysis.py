#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#sklearn imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from IPython.display import Markdown, display
from collections import defaultdict

#aif360 imports
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import StandardDataset
from aif360.explainers import MetricTextExplainer
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing


# In[ ]:


def fareclass_analysis(coupon_path, ticket_path,race_id_path):
    def aif360_dataframe(df):
        def majority(x,y):
            #x=white,y=non-white
            #This is used to find local majority of population
            #and return a 1 (white) or 0 (non-white)
            if x >= y:
                return 1.0
            elif y>x:
                return 0.0
            else:
                return 1.0 
    
        ls = ["Y", "C", "D"]
        def fare_classes(x):
            #filter fare class based on favorable and non-favorable 
            if x in ls: 
                return 1.0
            else:
                return 0.0
        #creating new columns
        fare_class_series= df["FareClass"].apply(lambda x: fare_classes(x)) 
        df=df.assign(fare_class=fare_class_series)
        df["local_majority_origin"]=df.apply(lambda x: majority(x["White alone proportion"],x["Non-White alone proportion"]),axis=1)

    
        #aif360 parameters
        label_name='fare_class'
        favorable_classes=[1.0]
        #using origin
        protected_attribute_names=['local_majority_origin']
        privileged_classes=[[1.0]]
        #example features,add or change features here,this is using origin
        features_keep=["fare_class","MilesFlown","local_majority_origin"] 
        #aif dataframe
        aif_data=StandardDataset(df,label_name,favorable_classes,protected_attribute_names,privileged_classes,features_to_keep=features_keep
                   ,metadata={'protected_attribute_maps': [{1.0: 'white', 0.0: 'non-white'}]})
        #more aif parameters
        sens_ind = 0
        sens_attr = aif_data.protected_attribute_names[sens_ind]

        unprivileged_groups = [{sens_attr: v} for v in
                               aif_data.unprivileged_protected_attributes[sens_ind]]
        privileged_groups = [{sens_attr: v} for v in
                             aif_data.privileged_protected_attributes[sens_ind]]
        metric_orig_panel = BinaryLabelDatasetMetric(
            StandardDataset(df,label_name,favorable_classes,protected_attribute_names,privileged_classes,features_to_keep=features_keep),
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
        
        explainer_orig_panel = MetricTextExplainer(metric_orig_panel)

        print(explainer_orig_panel.disparate_impact())
        return aif_data
    
    def describe_metrics(metrics, thresh_arr):
        best_ind = np.argmax(metrics['bal_acc'])
        print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
        print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
        disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
        print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
        print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
        print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
        print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
        print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))
        print("\n")
        
        
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
    
    
    
    #downloading data
    coupon1 = pd.read_csv(coupon_path)
    tickets1 = pd.read_csv(ticket_path)
    census = pd.read_csv(race_id_path) 
    
    #cleaning/preparing data 
    census = census[['Area Name', 'Metro Area', 'Code', 'White alone proportion', 'Non-White alone proportion']]
    
    #comninded both tickets and coupons 
    combinded = tickets1.merge(coupon1, left_on = "ItinID", right_on= "ItinID")
    #merging census 
    df = combinded.merge(census, left_on = 'OriginCityMarketID_y', right_on= 'Code')
    
    df = df[['Coupons_x', 'RoundTrip', 'OnLine', 'DollarCred', 'FarePerMile', 'Passengers_x', 
             'ItinFare', 'BulkFare', 'Distance_x', 'MilesFlown', 'FareClass',
             'White alone proportion','Non-White alone proportion']]
    
    df = df.dropna()
    
    #creating aif dataset
    aif_data = aif360_dataframe(df)
    (aif_train,aif_val,aif_test) = aif_data.split([0.5, 0.8], shuffle=True)
        
        
    #orginal data on random forest 
    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=20, min_samples_leaf=5))
    fit_params = {'randomforestclassifier__sample_weight': aif_train.instance_weights}
    rf_orig_panel = model.fit(aif_train.features,aif_train.labels.ravel(), **fit_params)
    
    
    sens_ind = 0
    sens_attr = aif_train.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           aif_train.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                         aif_train.privileged_protected_attributes[sens_ind]]
    

    
    
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_val,
                       model=model,
                       thresh_arr=thresh_arr)
    lr_simple_best_ind = np.argmax(val_metrics['bal_acc'])
    
    
    describe_metrics(val_metrics, thresh_arr)
    
    
    #applying reweighing
    
    # preprocssing reweighing 
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    dataset_transf_panel_train = RW.fit_transform(aif_train)
    
    
    sens_ind = 0
    
    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=20, min_samples_leaf=5))
    fit_params = {'randomforestclassifier__sample_weight': dataset_transf_panel_train.instance_weights}
    rf_trans_panel = model.fit(dataset_transf_panel_train.features,dataset_transf_panel_train.labels.ravel(), **fit_params)
    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=dataset_transf_panel_train,
                       model=model,
                       thresh_arr=thresh_arr)
    lr_simple_best_ind = np.argmax(val_metrics['bal_acc'])
    
    describe_metrics(val_metrics, thresh_arr)
    

