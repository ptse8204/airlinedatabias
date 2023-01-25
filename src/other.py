### Edwin
## Combined
# This function would provide a table that grab columns
# that we need in both sets and return a useful df

# Essential this table would combine the 2 tables with only useful columns

def gen_ticket_coupon(ticket_df, coupon_df):
  ticket_df_reduced = ticket_df[["ItinID", "Coupons", 'Year', 'Quarter', 
                                 'Origin', 'OriginCityMarketID', 'OriginState',
                                 'RoundTrip', 'OnLine', 'DollarCred', 'FarePerMile',
                                  'RPCarrier', 'Passengers', 'ItinFare', 'BulkFare'
                                  , 'MilesFlown', 'ItinGeoType']]
  del ticket_df
  coupon_df_reduced = coupon_df[['ItinID','SeqNum', 'Coupons', 'Year', 
                                 'Quarter', 'DestCityMarketID', 'Dest', 
                                 'DestState', 'CouponGeoType', 'FareClass']]
  del coupon_df
  max_gp = coupon_df_reduced[["SeqNum", "ItinID"]].groupby("ItinID").max().reset_index()
  coupon_df_filter = coupon_df_reduced.merge(max_gp, on=["ItinID",	"SeqNum"])
  return ticket_df_reduced.merge(coupon_df_filter, on=['ItinID', 'Year', 'Quarter'])

combined = gen_ticket_coupon(ticket, coupon)

### Edwin

## Connecting census dataset
# Reading Census City Data
def read_cen_data(path):
  census_city_code = pd.read_csv(path)
  census_city_code["median_income"] = census_city_code["11"].str.replace(",", "").astype("int")
  return census_city_code

# Just a histogram for median income
def us_city_median_income_plot(census_df):
  census_df.median_income.plot(kind="hist")

  ### Edwin
  ## Census EDA with Airline Data, Prereq: Combined, Census

  # This shows the data statistics of city areas that has 
  # median income in the bottom 25 percentile
  def bottom_25_data(combined, census_df):
    bot_25_median = census_df.median_income.quantile(.25)
    bottom_25_origins = combined[["FarePerMile", "RPCarrier", "OriginCityMarketID", 
                                  "MilesFlown", "SeqNum"]].merge(
      census_df[census_df.median_income <= bot_25_median][["Code", "median_income"]], 
      left_on = "OriginCityMarketID", right_on="Code")
    bottom_25_dest = combined[["FarePerMile", "RPCarrier", "DestCityMarketID", 
                               "MilesFlown", "SeqNum"]].merge(
      census_df[census_df.median_income <= bot_25_median][["Code", "median_income"]], 
      left_on = "DestCityMarketID", right_on="Code")
    print("Flights originate city areas that has median income in the bottom 25 percentile")
    print("Mean of FarePerMile : ", bottom_25_origins.FarePerMile.mean())
    print("Mean of MilesFlown : ", bottom_25_origins.MilesFlown.mean())
    print("Mean of Average Segments:", bottom_25_origins.SeqNum.mean())
    print("FarePerMile by carrier:")
    bottom_25_origins[["FarePerMile", "RPCarrier"]].groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")
    plt.show()
    print("FarePerMile Distribution:")
    bottom_25_origins.FarePerMile.hist()
    plt.show()
    print("Flights destination is city areas that has median income in the bottom 25 percentile")
    print("Mean of FarePerMile : ", bottom_25_dest.FarePerMile.mean())
    print("Mean of MilesFlown : ", bottom_25_dest.MilesFlown.mean())
    print("Mean of Average Segments:", bottom_25_dest.SeqNum.mean())
    print("FarePerMile by carrier:")
    bottom_25_dest[["FarePerMile", "RPCarrier"]].groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")
    plt.show()
    print("FarePerMile Distribution:")
    bottom_25_dest.FarePerMile.hist()
    plt.show()

  # This shows the data statistics of city areas that has 
  # median income in the upper 25 percentile
  def upper_25_data(combined, census_df):
    bot_25_median = census_df.median_income.quantile(.75)
    upper_25_origins = combined[["FarePerMile", "RPCarrier", "OriginCityMarketID", 
                                  "MilesFlown", "SeqNum"]].merge(
      census_df[census_df.median_income >= bot_25_median][["Code", "median_income"]], 
      left_on = "OriginCityMarketID", right_on="Code")
    upper_25_dest = combined[["FarePerMile", "RPCarrier", "DestCityMarketID", 
                               "MilesFlown", "SeqNum"]].merge(
      census_df[census_df.median_income >= bot_25_median][["Code", "median_income"]], 
      left_on = "DestCityMarketID", right_on="Code")
    print("Flights originate city areas that has median income in the upper 25 percentile")
    print("Mean of FarePerMile : ", upper_25_origins.FarePerMile.mean())
    print("Mean of MilesFlown : ", upper_25_origins.MilesFlown.mean())
    print("Mean of Average Segments:", upper_25_origins.SeqNum.mean())
    print("FarePerMile by carrier:")
    upper_25_origins[["FarePerMile", "RPCarrier"]].groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")
    plt.show()
    print("FarePerMile Distribution:")
    upper_25_origins.FarePerMile.hist()
    plt.show()
    print("Flights destination is city areas that has median income in the upper 25 percentile")
    print("Mean of FarePerMile : ", upper_25_dest.FarePerMile.mean())
    print("Mean of MilesFlown : ", upper_25_dest.MilesFlown.mean())
    print("Mean of Average Segments:", upper_25_dest.SeqNum.mean())
    print("FarePerMile by carrier:")
    upper_25_dest[["FarePerMile", "RPCarrier"]].groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")
    plt.show()
    print("FarePerMile Distribution:")
    upper_25_dest.FarePerMile.hist()
    plt.show()


  # This shows the flight statistics of city areas that has 
  # with upper and lower 25th percentile as orgin and dest
  # and vice-versa
  def lower_and_upper_data(combined, census_df):
    bot_25_median = census_df.median_income.quantile(.25)
    up_25_median = census_df.median_income.quantile(.75)
    origins = combined[["FarePerMile", "RPCarrier", "OriginCityMarketID", 
                                  "DestCityMarketID", "MilesFlown", "SeqNum"]].merge(
      census_df[census_df.median_income <= bot_25_median][["Code", "median_income"]], 
      left_on = "OriginCityMarketID", right_on="Code")
    origin_dest = origins.merge(
      census_df[census_df.median_income >= up_25_median][["Code", "median_income"]], 
      left_on = "DestCityMarketID", right_on="Code")
    print("Flights originate bottom 25 to upper 25")
    print("Mean of FarePerMile : ", origin_dest.FarePerMile.mean())
    print("Mean of MilesFlown : ", origin_dest.MilesFlown.mean())
    print("Mean of Average Segments:", origin_dest.SeqNum.mean())
    print("FarePerMile by carrier:")
    origin_dest[["FarePerMile", "RPCarrier"]].groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")
    plt.show()
    print("FarePerMile Distribution:")
    origin_dest.FarePerMile.hist()
    plt.show()
    origins = combined[["FarePerMile", "RPCarrier", "OriginCityMarketID", 
                                  "DestCityMarketID", "MilesFlown", "SeqNum"]].merge(
      census_df[census_df.median_income >= up_25_median][["Code", "median_income"]], 
      left_on = "OriginCityMarketID", right_on="Code")
    origin_dest = origins.merge(
      census_df[census_df.median_income <= bot_25_median][["Code", "median_income"]], 
      left_on = "DestCityMarketID", right_on="Code")
    print("Flights originate upper 25 to bottom 25")
    print("Mean of FarePerMile : ", origin_dest.FarePerMile.mean())
    print("Mean of MilesFlown : ", origin_dest.MilesFlown.mean())
    print("Mean of Average Segments:", origin_dest.SeqNum.mean())
    print("FarePerMile by carrier:")
    origin_dest[["FarePerMile", "RPCarrier"]].groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")
    plt.show()
    print("FarePerMile Distribution:")
    origin_dest.FarePerMile.hist()
    plt.show()

  # This shows the flight statistics of city areas that has 
  # with upper25 as both orgin and dest
  # and lower as both orgin and dest
  def double_low_high(combined, census_df):
    bot_25_median = census_df.median_income.quantile(.25)
    up_25_median = census_df.median_income.quantile(.75)
    origins = combined[["FarePerMile", "RPCarrier", "OriginCityMarketID", 
                                  "DestCityMarketID", "MilesFlown", "SeqNum"]].merge(
      census_df[census_df.median_income <= bot_25_median][["Code", "median_income"]], 
      left_on = "OriginCityMarketID", right_on="Code")
    origin_dest = origins.merge(
      census_df[census_df.median_income <= bot_25_median][["Code", "median_income"]], 
      left_on = "DestCityMarketID", right_on="Code")
    print("Flights originate and destin for bottom 25")
    print("Mean of FarePerMile : ", origin_dest.FarePerMile.mean())
    print("Mean of MilesFlown : ", origin_dest.MilesFlown.mean())
    print("Mean of Average Segments:", origin_dest.SeqNum.mean())
    print("FarePerMile by carrier:")
    origin_dest[["FarePerMile", "RPCarrier"]].groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")
    plt.show()
    print("FarePerMile Distribution:")
    origin_dest.FarePerMile.hist()
    plt.show()
    origins = combined[["FarePerMile", "RPCarrier", "OriginCityMarketID", 
                                  "DestCityMarketID", "MilesFlown", "SeqNum"]].merge(
      census_df[census_df.median_income >= up_25_median][["Code", "median_income"]], 
      left_on = "OriginCityMarketID", right_on="Code")
    origin_dest = origins.merge(
      census_df[census_df.median_income >= up_25_median][["Code", "median_income"]], 
      left_on = "DestCityMarketID", right_on="Code")
    print("Flights originate and destin for upper 25")
    print("Mean of FarePerMile : ", origin_dest.FarePerMile.mean())
    print("Mean of MilesFlown : ", origin_dest.MilesFlown.mean())
    print("Mean of Average Segments:", origin_dest.SeqNum.mean())
    print("FarePerMile by carrier:")
    origin_dest[["FarePerMile", "RPCarrier"]].groupby("RPCarrier").mean()["FarePerMile"].plot(kind="bar")
    plt.show()
    print("FarePerMile Distribution:")
    origin_dest.FarePerMile.hist()
    plt.show()
