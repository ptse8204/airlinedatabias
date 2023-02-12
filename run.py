#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import src.bias_mitigation_analysis

if __name__ == "__main__":
    if len(sys.argv)>1:
        x = sys.argv[1]
        if x =="test":
            #this needs to use the test data instead
            print("test data")
        else:
            ticket=pd.read_csv("Origin_and_Destination_Survey_DB1BTicket_2022_1.csv",nrows=1000000)
            income=pd.read_csv("median_income_cityID.csv")
            race=pd.read_csv("race_id.csv")
            market=pd.read_csv("Origin_and_Destination_Survey_DB1BMarket_2022_1.csv",nrows=1000000)    
            src.bias_mitigation_analysis.bias_mitigation(ticket,market, race, income)
    else:
        ticket=pd.read_csv("Origin_and_Destination_Survey_DB1BTicket_2022_1.csv",nrows=1000000)
        income=pd.read_csv("median_income_cityID.csv")
        race=pd.read_csv("race_id.csv")
        market=pd.read_csv("Origin_and_Destination_Survey_DB1BMarket_2022_1.csv",nrows=1000000)    
        src.bias_mitigation_analysis.bias_mitigation(ticket,market, race, income)
