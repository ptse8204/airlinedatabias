#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import src.farepermile_biasmitigation
import src.fareclass_analysis
import src.preliminarypricesensitivitymodel

if __name__ == "__main__":
    if len(sys.argv)>1:
        x = sys.argv[1]
        if x =="test":
            #this needs to use the test data instead
            print("test data")
            test=True
            print("Bias Analysis")
            ticket=pd.read_csv("test/testdata/test_ticket.csv",nrows=1000000)
            income=pd.read_csv("test/testdata/test_income.csv")
            race=pd.read_csv("test/testdata/test_race.csv")
            market=pd.read_csv("test/testdata/test_market.csv",nrows=1000000) 
            coupon=pd.read_csv("test/testdata/test_coupon.csv",nrows=1000000)
            src.farepermile_biasmitigation.fpm_bias_mitigation(ticket,coupon,race,test)
            print("Fareclass Model")
            src.fareclass_analysis.fareclass_analysis(coupon, ticket,race)
            print("Price Sensitivity Model")
            src.preliminarypricesensitivitymodel.price_sensitivity_model(ticket,coupon,test)
        else:
            test=False
            print("Bias Analysis")
            ticket=pd.read_csv("Origin_and_Destination_Survey_DB1BTicket_2022_1.csv",nrows=1000000)
            income=pd.read_csv("median_income_cityID.csv")
            race=pd.read_csv("race_id.csv")
            market=pd.read_csv("Origin_and_Destination_Survey_DB1BMarket_2022_1.csv",nrows=1000000)    
            src.farepermile_biasmitigation.fpm_bias_mitigation(ticket,coupon,race,test)
            print("Fareclass Model")
            src.fareclass_analysis.fareclass_analysis(coupon, ticket,race)
            print("Price Sensitivity Model")
            src.preliminarypricesensitivitymodel.price_sensitivity_model(ticket,coupon,test)
    else:
        test=False
        print("Bias Analysis")
        ticket=pd.read_csv("Origin_and_Destination_Survey_DB1BTicket_2022_1.csv",nrows=1000000)
        income=pd.read_csv("median_income_cityID.csv")
        race=pd.read_csv("race_id.csv")
        market=pd.read_csv("Origin_and_Destination_Survey_DB1BMarket_2022_1.csv",nrows=1000000)    
        src.farepermile_biasmitigation.fpm_bias_mitigation(ticket,coupon,race,test)
        print("Fareclass Model")
        src.fareclass_analysis.fareclass_analysis(coupon, ticket,race)
        print("Price Sensitivity Model")
        src.preliminarypricesensitivitymodel.price_sensitivity_model(ticket,coupon,test)
