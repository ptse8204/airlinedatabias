# This is the preliminary model for the price sensitivity model

## Importing imports
import pandas as pd
import coupon, graber, manager, other, t100, ticket
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor ##
import pickle ##

## Importing datasets and data wrangling
year_used = 2018
q_used = 2
ticket_df_1 = manager.load_ticket_df("/content/drive/MyDrive/UCSD/DSC180B/Airline Data/Ticket", year_used, q_used)
coupon_df_1 = manager.load_coupon_df("/content/drive/MyDrive/UCSD/DSC180B/Airline Data/Coupon", year_used, q_used)
combine_1 = t100.t100_combined(ticket_df_1, coupon_df_1, 
                               t100.import_T100_by_yr("/content/drive/MyDrive/UCSD/DSC180B/T-100/Segment/", year_used))

# ensuring memory would not explode                           
del ticket_df_1
del coupon_df_1

filtered_combine = other.dropping_outlier(combine_1)
del combine_1

f_c_nna = filtered_combine[~filtered_combine.LOAD_FACTOR.isna()]
# generate cdf numbers
f_c_nna["cdf"] = f_c_nna.groupby[["OriginCityMarketIDCoupon", "DestCityMarketID", "RPCarrier",
                  "ItinFare"]].groupby([
                      "OriginCityMarketIDCoupon", "DestCityMarketID", "RPCarrier"]
                      ).rank(pct=True)
airline_cdf = f_c_nna[[
"OriginCityMarketIDCoupon",	"DestCityMarketID",	"RPCarrier",	"FarePerMile",	"CouponDistance",	"MilesFlown",	"RoundTrip",	"LOAD_FACTOR",	"cdf"]]

## Splitting train and test sets
X_train, X_test, y_train,y_test = train_test_split(airline_cdf.iloc[:, :-1], airline_cdf.iloc[:, -1])

lgbm = LGBMRegressor(n_jobs=-1, warm_start=True)
scaler = StandardScaler()
encode = OneHotEncoder(handle_unknown='ignore')
lgb_regs = Pipeline([("preprocess", ColumnTransformer([
     ('categorical_preprocessing', encode, [ 
                     "OriginCityMarketIDCoupon", "DestCityMarketID",
                      "RPCarrier", "RoundTrip"]),
    ('numerical_preprocessing', scaler, ["FarePerMile"
                            , "CouponDistance", "MilesFlown", "LOAD_FACTOR"])
],sparse_threshold=0)), 
("quan_reg", lgbm)])

lgb_regs.fit(X_train.iloc[:round(5217289*.05)], y_train.iloc[:round(5217289*.05)])

# save the model to disk
filename = 'lgb_regs.sav'
pickle.dump(lgb_regs, open(filename, 'wb'))
 
