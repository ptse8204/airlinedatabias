import pandas as pd
import numpy as np
import random

from collections import defaultdict
from aif360.datasets import StandardDataset
from aif360.datasets import BinaryLabelDataset
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline 
from aif360.explainers import MetricTextExplainer
from sklearn.preprocessing import StandardScaler
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# Test is True or False depending on whether using test set (True) or not (False)
def fpm_bias_mitigation(ticket, coupon, race, test):
    def combined_based_coupon(ticket_df, coupon_df):
        ticket_df_reduced = ticket_df[["ItinID", "Coupons", 'Year', 'Quarter', 
                                      'Origin', 'OriginCityMarketID', 'OriginState',
                                      'RoundTrip', 'OnLine', 'DollarCred', 'FarePerMile', 
                                        'RPCarrier', 'Passengers', 'ItinFare', 'BulkFare'
                                        , 'MilesFlown', 'DistanceGroup', 'ItinGeoType']]
        del ticket_df
        coupon_df_reduced = coupon_df[['ItinID','SeqNum', 'Coupons', 'Year', 
                                      'Quarter', 'DestCityMarketID', 'Dest', 
                                      'DestState', 'CouponGeoType', 'FareClass','Distance',
                                     'DistanceGroup']].rename(columns={'Distance': 'CouponDistance',
                                     'DistanceGroup': 'CouponDistanceGroup'})
        del coupon_df
        return ticket_df_reduced.merge(coupon_df_reduced, on=['ItinID', 'Year', 'Quarter', 'Coupons'], how="left")

    df = combined_based_coupon(ticket, coupon)
    df = df.groupby(["ItinID", "Coupons", 'Year', 'Quarter', 
          'Origin', 'OriginCityMarketID', 'OriginState',
          'RoundTrip', 'OnLine', 'DollarCred', 'FarePerMile', 
            'RPCarrier', 'Passengers', 'ItinFare', 'BulkFare', 
         'MilesFlown', 'DistanceGroup', 'ItinGeoType']).agg({"DestCityMarketID": list}).reset_index()
    df["Flight Path"] = df.apply(lambda x: str([x["OriginCityMarketID"]] + x["DestCityMarketID"]), axis=1)
    df["LastCityMarketID"] = df["DestCityMarketID"].apply(lambda x: x[len(x)-1])
    counts = df["Flight Path"].value_counts()
    if not test:
        counts = counts[counts >= 50]
    df = df[df["Flight Path"].isin(counts.index)]
    df = df.groupby("Flight Path").agg({"FarePerMile":np.median, 'RoundTrip':pd.Series.mode, 'OnLine':np.mean, 
                               'OriginCityMarketID': pd.Series.mode, "LastCityMarketID": pd.Series.mode, 
                               "RPCarrier": pd.Series.mode, "DistanceGroup":pd.Series.mode, "ItinGeoType":pd.Series.mode}).reset_index()
    df = df.merge(race, left_on="OriginCityMarketID", right_on="Code")
    df = df.merge(race, left_on="LastCityMarketID", right_on="Code")

    def simulate_race(row):
        if random.uniform(0, 1) < (row["White alone proportion_x"] + row["White alone proportion_y"]) / 2:
            return 1;
        return 0;

    def majority_race(row):
        if 0.5 <= (row["White alone proportion_x"] + row["White alone proportion_y"]) / 2:
            return 1;
        return 0;

    df["Race"] = df.apply(majority_race, axis=1)
    df["DistanceGroup"] = df["DistanceGroup"].apply(lambda x: x if isinstance(x, int) else np.mean(x))
    df["RPCarrier"] = df["RPCarrier"].apply(lambda x: x if isinstance(x, str) else str(x))
    
    def describe_metrics(metrics, thresh_arr):
        best_ind = np.argmax(metrics['bal_acc'])
        print("Threshold corresponding to Best balanced accuracy: {:6.4f}".format(thresh_arr[best_ind]))
        print("Best balanced accuracy: {:6.4f}".format(metrics['bal_acc'][best_ind]))
    #     disp_imp_at_best_ind = np.abs(1 - np.array(metrics['disp_imp']))[best_ind]
        disp_imp_at_best_ind = 1 - min(metrics['disp_imp'][best_ind], 1/metrics['disp_imp'][best_ind])
        print("Corresponding Disparate Impact: {:6.4f}".format(metrics['disp_imp'][best_ind]))
        print("Corresponding 1-min(DI, 1/DI) value: {:6.4f}".format(disp_imp_at_best_ind))
        print("Corresponding average odds difference value: {:6.4f}".format(metrics['avg_odds_diff'][best_ind]))
        print("Corresponding statistical parity difference value: {:6.4f}".format(metrics['stat_par_diff'][best_ind]))
        print("Corresponding equal opportunity difference value: {:6.4f}".format(metrics['eq_opp_diff'][best_ind]))
        print("Corresponding Theil index value: {:6.4f}".format(metrics['theil_ind'][best_ind]))


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
    
    percent25 = df["FarePerMile"].describe()["75%"]
    fare_class_series=df["FarePerMile"].apply(lambda x: x < percent25)
    df=df.assign(fare_class=fare_class_series)

    label_name='fare_class'
    favorable_classes=[1.0]

    protected_attribute_names=['Race']
    privileged_classes=[[1.0]]

    features_keep=["RoundTrip","OnLine","DistanceGroup"]
    categorical_features=["OriginCityMarketID","LastCityMarketID","RPCarrier","ItinGeoType"]
    #aif dataframe
    aif_data=StandardDataset(df,label_name,favorable_classes,protected_attribute_names,privileged_classes,categorical_features=categorical_features,
                             features_to_keep=features_keep, metadata={'protected_attribute_maps': [{1.0: 'white', 0.0: 'non-white'}]})

    sens_ind = 0
    sens_attr = aif_data.protected_attribute_names[sens_ind]

    unprivileged_groups = [{sens_attr: v} for v in
                           aif_data.unprivileged_protected_attributes[sens_ind]]
    privileged_groups = [{sens_attr: v} for v in
                           aif_data.privileged_protected_attributes[sens_ind]]

    (aif_train,
     aif_val,
     aif_test) = aif_data.split([0.5, 0.8], shuffle=True)


    metric_train = BinaryLabelDatasetMetric(
            aif_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
    explainer_train = MetricTextExplainer(metric_train)
    #explainer_train_lst.append(explainer_train)

    # train
    dataset = aif_train
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

    lr_model = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_val,
                       model=lr_model,
                       thresh_arr=thresh_arr)
    lr_best_ind = np.argmax(val_metrics['bal_acc'])

    lr_metrics = test(dataset=aif_test,
                       model=lr_model,
                       thresh_arr=[thresh_arr[lr_best_ind]])

    print("Logistic Regression: No Bias Mitigation")
    describe_metrics(lr_metrics, [thresh_arr[lr_best_ind]])
    
    (aif_train,
     aif_val,
     aif_test) = aif_data.split([0.5, 0.8], shuffle=True)


    metric_train = BinaryLabelDatasetMetric(
            aif_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
    explainer_train = MetricTextExplainer(metric_train)
    #explainer_train_lst.append(explainer_train)

    # train
    dataset = aif_train
    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
    fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}

    rf_model = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_val,
                       model=rf_model,
                       thresh_arr=thresh_arr)
    rf_best_ind = np.argmax(val_metrics['bal_acc'])

    rf_metrics = test(dataset=aif_test,
                       model=rf_model,
                       thresh_arr=[thresh_arr[rf_best_ind]])
    
    print("Random Forest: No Bias Mitigation")
    describe_metrics(rf_metrics, [thresh_arr[rf_best_ind]])

    aif_data=StandardDataset(df,label_name,favorable_classes,protected_attribute_names,privileged_classes,categorical_features=categorical_features,
                         features_to_keep=features_keep, metadata={'protected_attribute_maps': [{1.0: 'white', 0.0: 'non-white'}]})
    RW = Reweighing(unprivileged_groups=unprivileged_groups,
                    privileged_groups=privileged_groups)
    aif_data = RW.fit_transform(aif_data)
    
    (aif_train,
     aif_val,
     aif_test) = aif_data.split([0.5, 0.8], shuffle=True)


    metric_train = BinaryLabelDatasetMetric(
            aif_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
    explainer_train = MetricTextExplainer(metric_train)
    #explainer_train_lst.append(explainer_train)

    # train
    dataset = aif_train
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(solver='liblinear', random_state=1))
    fit_params = {'logisticregression__sample_weight': dataset.instance_weights}

    lr_rw_model = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_val,
                       model=lr_rw_model,
                       thresh_arr=thresh_arr)
    lr_rw_best_ind = np.argmax(val_metrics['bal_acc'])

    lr_rw_metrics = test(dataset=aif_test,
                       model=lr_rw_model,
                       thresh_arr=[thresh_arr[lr_rw_best_ind]])
    
    print("Logistic Regression: Reweighing")
    describe_metrics(lr_rw_metrics, [thresh_arr[lr_rw_best_ind]])
    
    (aif_train,
     aif_val,
     aif_test) = aif_data.split([0.5, 0.8], shuffle=True)


    metric_train = BinaryLabelDatasetMetric(
            aif_train,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups)
    explainer_train = MetricTextExplainer(metric_train)
    #explainer_train_lst.append(explainer_train)

    # train
    dataset = aif_train
    model = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=500, min_samples_leaf=25))
    fit_params = {'randomforestclassifier__sample_weight': dataset.instance_weights}

    rf_rw_model = model.fit(dataset.features, dataset.labels.ravel(), **fit_params)

    thresh_arr = np.linspace(0.01, 0.5, 50)
    val_metrics = test(dataset=aif_val,
                       model=lr_model,
                       thresh_arr=thresh_arr)
    rf_rw_best_ind = np.argmax(val_metrics['bal_acc'])

    rf_rw_metrics = test(dataset=aif_test,
                       model=rf_rw_model,
                       thresh_arr=[thresh_arr[rf_rw_best_ind]])

    print("Random Forest: Reweighing")
    describe_metrics(rf_rw_metrics, [thresh_arr[rf_rw_best_ind]])
    
    aif_data=StandardDataset(df,label_name,favorable_classes,protected_attribute_names,privileged_classes,categorical_features=categorical_features,
                         features_to_keep=features_keep, metadata={'protected_attribute_maps': [{1.0: 'white', 0.0: 'non-white'}]})

    if test==True:
         (aif_train,
         aif_val,
         aif_test) = aif_data.split([0.01, 0.7], shuffle=True)
    else:
        (aif_train,
         aif_val,
         aif_test) = aif_data.split([0.5, 0.8], shuffle=True)

    model = PrejudiceRemover(sensitive_attr=sens_attr)
    pr_orig_scaler = StandardScaler()

    dataset = aif_train.copy()
    dataset.features = pr_orig_scaler.fit_transform(dataset.features)

    pr_orig_model = model.fit(dataset)

    thresh_arr = np.linspace(0.01, 0.50, 50)

    dataset = aif_val.copy()
    dataset.features = pr_orig_scaler.transform(dataset.features)

    val_metrics = test(dataset=dataset,
                       model=pr_orig_model,
                       thresh_arr=thresh_arr)
    pr_orig_best_ind = np.argmax(val_metrics['bal_acc'])


    dataset = aif_test.copy()
    dataset.features = pr_orig_scaler.transform(dataset.features)

    pr_orig_metrics = test(dataset=dataset,
                           model=pr_orig_model,
                           thresh_arr=[thresh_arr[pr_orig_best_ind]])
    
    print("Prejudice Remover")
    describe_metrics(pr_orig_metrics, [thresh_arr[pr_orig_best_ind]])
    
    results = [lr_metrics, rf_metrics, lr_rw_metrics,
           rf_rw_metrics, pr_orig_metrics]
    debias = pd.Series(['']*2 + ['Reweighing']*2
                     + ['Prejudice Remover'],
                       name='Bias Mitigator')
    clf = pd.Series(['Logistic Regression', 'Random Forest']*2 + [''],
                    name='Classifier')
    return pd.concat([pd.DataFrame(metrics) for metrics in results], axis=0).set_index([debias, clf])
