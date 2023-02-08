
## T-100
# Please only use the domestic segment data for this module

# dependency
import pandas as pd
import other

# would return only with flights that are schedule for passenger
def import_T100(path):
  segment = pd.read_csv(path)
  print("Successfully imported!")
  # filtering with passenger class, passenger aircraft, and flights that never run
  passenger = segment[(segment.CLASS == "F") & (segment["AIRCRAFT_CONFIG"] == 1) & segment["DEPARTURES_PERFORMED"] != 0]
  return passenger

# A preset of useful columns
def useful_cols(passenger):
  return passenger[["DEPARTURES_PERFORMED", "CARRIER", 'UNIQUE_CARRIER_NAME','SEATS','PASSENGERS'
                    , 'DISTANCE', 'RAMP_TO_RAMP', 'CARRIER_GROUP_NEW', 'ORIGIN_CITY_MARKET_ID', 
                'ORIGIN', 'DEST_CITY_MARKET_ID', 'DEST', 'YEAR', 'QUARTER', 'MONTH', 'DISTANCE_GROUP']]

# Grabbing load factor value series
# note that the orginal dataset passenger and load count a summarized with the same "departure"
# this means that if depart_per = 100, passenger = 10000, per flight passenger avg is 100
def gen_load_factor_series(t100_df):
  return t100_df.PASSENGERS / t100_df.SEATS / t100_df.DEPARTURES_PERFORMED

# t-100 segment should only be merging with coupon as it is per segement
# note that t-100 is annual data and coupon is better be use as a quarter data
# due to its sear size
# note that this would only match with market city pair and give the coupon dataset
# the load factor and departures performed 
# the function would run and grab the load_factor variable and should be 
# able to run with load_factor var in with no error as well
def matching_coupon(t100_df, coupon_df):
  t100_df_load = t100_df.copy()
  t100_df_load["LOAD_FACTOR"] = gen_load_factor_series(t100_df_load)
  t_100_grouped = t_100_df.groupby(['YEAR', 'QUARTER', 'CARRIER','ORIGIN_CITY_MARKET_ID', 
                                    'DEST_CITY_MARKET_ID'])
  # this is the city-pair load factor given on a specific carrier
  t_100_grouped_mean = t_100_grouped.mean()[["LOAD_FACTOR"]]
  # The pass_tran and set_tran means given year, quarter and city-pair the total amount of such available
  t_100_grouped_sum = t_100_grouped.sum()[["PASSENGERS", 'SEATS', 
                                           "DEPARTURES_PERFORMED"]].rename(
                                               columns={"PASSENGERS": "PASSENGERS_TRANS", 'SEATS': "SEATS_TRANS"})
  t_100_stat = t_100_grouped_mean.join(t_100_grouped_sum).reset_index()
  return coupon_df.merge(t_100_stat, how="left", left_on = ['Year', 
                                  'Quarter', 'OpCarrier','OriginCityMarketID', 
                                   DestCityMarketID'], right_on = ['YEAR', 
                                  'QUARTER', 'CARRIER','ORIGIN_CITY_MARKET_ID', 
                                    'DEST_CITY_MARKET_ID'])

# t-100 w/combined
def t100_combined(ticket_df, coupon_df, t100_path):
  ticket_df_reduced = ticket_df[["ItinID", "Coupons", 'Year', 'Quarter', 
                                  'Origin', 'OriginCityMarketID', 'OriginState',
                                  'RoundTrip', 'OnLine', 'DollarCred', 'FarePerMile',
                                    'RPCarrier', 'Passengers', 'ItinFare', 'BulkFare'
                                    , 'MilesFlown', 'ItinGeoType']].rename(
                                        columns={"Coupons": "TotalCouponCount"})
  del ticket_df
  pre_t100 = coupon_df[['Year','Quarter', 'OpCarrier','OriginCityMarketID', 
                        DestCityMarketID','ItinID','SeqNum', 'Coupons','Dest', 
                                  'DestState', 'CouponGeoType', 'FareClass','Distance',
                                 'DistanceGroup']]
  del coupon_df
  coupon_t100 = matching_coupon(useful_cols(import_T100(t100_path)), pre_t100)
  del pre_t100
  coupon_df_reduced = coupon_t100[['ItinID','SeqNum', 'Coupons', 'Year', 
                                  'Quarter', 'DestCityMarketID', 'Dest', 
                                  'DestState', 'CouponGeoType', 'FareClass','Distance',
                                 'DistanceGroup', "PASSENGERS_TRANS", "SEATS_TRANS",
                                 "DEPARTURES_PERFORMED", "LOAD_FACTOR"]].rename(
                                   columns={'Distance': 'CouponDistance',
                                 'DistanceGroup': 'CouponDistanceGroup'})
  del coupon_t100
  return ticket_df_reduced.merge(coupon_df_reduced, on=['ItinID', 'Year', 'Quarter'], how="right")
