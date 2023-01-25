# Dependency
import pandas as pd
import numpy as np


# Name
# and dump your function

#Joseph
def flight_length_bar(dataset):
  #Flight length
  def flight_length(x):
      if x <= 1725: #miles
          return "short-haul"
      elif x>1725 and x<=3450:
          return "medium-haul"
      else:
          return "long-haul"
  #adding new column to data and plotting bar chart
  dataset["flight_length"]=dataset["MilesFlown"].apply(lambda x: flight_length(x))
  lengths=dataset.groupby("flight_length").count()["ItinID"]
  plt.bar(x=lengths.index,height=lengths)
  plt.title("Flight Length Count")

#adjusting fare columns for inflation 
def inflation(dataset,inflation_amount):
    #inflation amount in decimals
    #ex: 2019 to 2022 = 14.5% ->0.145
    dataset["FarePerMile"]=dataset["FarePerMile"].apply(lambda x: x + (x*inflation_amount)) 
    dataset["ItinFare"]=dataset["ItinFare"].apply(lambda x: x + (x*inflation_amount)) 
    return dataset

### Qixi Huang
### $1 in 2018 is worth $1.16 in 2022
def adjust_inflation(dataset):
    dataset['ItinFare'] = dataset['ItinFare'].apply(lambda x: x * 1.16)
    dataset['FarePerMile'] = dataset['FarePerMile'].apply(lambda x: x * 1.16)

def Market_share_Carrier(dataset):
    ### plot bars chart that show market share for each airlines
    carrier_percent = dataset.RPCarrier.value_counts()
    carrier_percent.plot.bar()

def Rev_Carrier(dataset):
    ### plot bars chart that show revenue per miles on each airlines
    avg_rev = dataset.groupby('RPCarrier')['FarePerMile'].mean()
    avg_rev.sort_values().plot.bar()

def fare_to_dis(dataset):
    ### bar charts that show fare per mile with respect to flight_length
    avg_rev_dis = dataset.groupby('flight_length')['FarePerMile'].mean()
    avg_rev_dis.plot.bar()

def carrier_option_dis(dataset):
    
    ### returns two dictionaries that shows whats the most and least profitable option
    ### in terms of flight length category, e.g. short hual maybe more profitable for some carriers.
    carrier_dis_table = dataset.groupby(['RPCarrier','flight_length'])['FarePerMile'].mean()
    
    carrier_percent = dataset.RPCarrier.value_counts()
    
    profit_dis_airline = dict()
    not_profit_dis_airline = dict()
    
    for i in carrier_percent.index:
        
        profit_dis_airline[i] = carrier_dis_table[i].idxmax()
        not_profit_dis_airline[i] = carrier_dis_table[i].idxmin()
        
    return [profit_dis_airline, not_profit_dis_airline]

# Garrick Su

import cpi 
from datetime import date

def inflation(row):
    month = 1 if row["Quarter"] == 1 else (3 if row["Quarter"] == 2 else (6 if row["Quarter"] == 3 else 9))
    row["ItinFare"] = cpi.inflate(row["ItinFare"], date(row["Year"], month, 1), to=date(2022, 1, 1))
    row["FarePerMile"] = cpi.inflate(row["FarePerMile"], date(row["Year"], month, 1), to=date(2022, 1, 1))
    return row

def convert_inflation(dataset):
    return dataset.apply(inflation, axis=1)

def clean_and_merge_race_city(L_CITY_MARKET_ID, race, dataset):
    race = race.T
    race.columns = race.iloc[0]
    race = race.drop(race.index[0]).reset_index()
    race["Metro Area"] = race['index'].apply(lambda x: x[-10:] == "Metro Area")
    race["Area Name"] = race['index'].apply(lambda x: x[:-11])
    race = race.merge(L_CITY_MARKET_ID, left_on="Area Name", right_on="Description", how='inner')
    return race, race.merge(dataset, left_on="Code", right_on="OriginCityMarketID", how='inner')

def lowest_and_highest_5(merged_dataset, merged_race):
    lowest_5 = merged_dataset.groupby("Code").mean()["ItinFare"].sort_values().iloc[0:5].index
    highest_5 = merged_dataset.groupby("Code").mean()["ItinFare"].sort_values().iloc[-5:].index
    return race[race["Code"].isin(lowest_5)], race[race["Code"].isin(highest_5)]
    
### Edwin
## Ticket, works with combined
# Drop fpm outliers
def filter_ticket_df_outliers(ticket_df):
  return ticket_df[combined["FarePerMile"] < ticket_df["FarePerMile"].quantile(.99)]

# This function return average FarePerMile per Carrier
def avg_fpm(ticket_df):
  ticket_df.groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")

# This function return average FarePerMile per Carrier for Legacy Airlines
def avg_fpm_legacy(ticket_df):
  ticket_df.groupby("RPCarrier").mean()["FarePerMile"].loc[[
      "AA", "AS", "B6", "DL", "HA", "UA", "WN"]].plot(kind="barh")
