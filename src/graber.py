
import pandas as pd

# getting lookup table for census data
def city_code_reader():
  return pd.read_csv("https://raw.githubusercontent.com/ptse8204/airlinedatabias/main/lookup/city_id_census.csv")

### Qixi (Jason)
### This function returns a table, which contains statistics of:
### 1. market share of each flight length category condition 
### on percentile of white population ratio in communities.
### 2. Avg Fare groupby by flight length and segmented 
### on percentile of white population ratio in communities.

def race_distance_df(combined, race_df):
    
    bot_25_median = race_df.white.quantile(.25)
    bottom_25_origins = combined[["FarePerMile", "RPCarrier", "OriginCityMarketID", 
                          "MilesFlown", "SeqNum",'Flight_length','ItinFare']].merge(
    race_df[race_df.white <= bot_25_median][["Code", "white"]], 
    left_on = "OriginCityMarketID", right_on="Code")
    bottom_25_dest = combined[["FarePerMile", "RPCarrier", "DestCityMarketID", 
                       "MilesFlown", "SeqNum",'Flight_length','ItinFare']].merge(
    race_df[race_df.white <= bot_25_median][["Code", "white"]], 
    left_on = "DestCityMarketID", right_on="Code")
    
    up_25_median = race_df.white.quantile(.75)
    up_25_origins = combined[["FarePerMile", "RPCarrier", "OriginCityMarketID", 
                          "MilesFlown", "SeqNum",'Flight_length','ItinFare']].merge(
    race_df[race_df.white >= up_25_median][["Code", "white"]], 
    left_on = "OriginCityMarketID", right_on="Code")
    up_25_dest = combined[["FarePerMile", "RPCarrier", "DestCityMarketID", 
                       "MilesFlown", "SeqNum",'Flight_length','ItinFare']].merge(
    race_df[race_df.white >= up_25_median][["Code", "white"]], 
    left_on = "DestCityMarketID", right_on="Code")
    
    
    result_df = pd.DataFrame()
    
    ### 1. market share of each flight length category condition 
    ### on percentile of white population ratio in communities.
    result_df["b25_dest_percent"] = \
    bottom_25_dest.Flight_length.value_counts(normalize = True).apply(lambda x: str(round(x*100,2)) +'%')
    result_df["b25_origin_percent"] = \
    bottom_25_origins.Flight_length.value_counts(normalize = True).apply(lambda x: str(round(x*100,2)) +'%')
    result_df["u25_dest_percent"] = \
    up_25_dest.Flight_length.value_counts(normalize = True).apply(lambda x: str(round(x*100,2)) +'%')
    result_df["u25_origin_percent"] = \
    up_25_origins.Flight_length.value_counts(normalize = True).apply(lambda x: str(round(x*100,2)) +'%')
    
    
    ### Avg Fare groupby by flight length and segmented 
    ### on percentile of white population ratio in communities.
    
    result_df["b25_dest_fare"] = \
    bottom_25_dest.groupby('Flight_length')['ItinFare'].mean().apply(lambda x: str(round(x,3)) + '$')
    result_df["b25_origin_fare"] = \
    bottom_25_origins.groupby('Flight_length')['ItinFare'].mean().apply(lambda x: str(round(x,3)) + '$')
    result_df["u25_dest_fare"] = \
    up_25_dest.groupby('Flight_length')['ItinFare'].mean().apply(lambda x: str(round(x,3)) + '$')
    result_df["u25_origin_fare"] = \
    up_25_origins.groupby('Flight_length')['ItinFare'].mean().apply(lambda x: str(round(x,3)) + '$')
    
    
    return result_df
