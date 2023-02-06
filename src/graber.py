
import pandas as pd

# getting lookup table for census data
def city_code_reader():
  return pd.read_csv("https://raw.githubusercontent.com/ptse8204/airlinedatabias/main/lookup/city_id_census.csv")
