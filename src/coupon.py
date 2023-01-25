### Edwin
## Coupon, works with combined
# This function allows filtering different classes
# accept input of a single class
def select_class(coupon_df, class_str):
  return coupon_df[coupon_df['FareClass'] == class_str]

# This function allows filtering only the class (X, Y)
def econ_class(coupon_df):
  return coupon_df[(coupon_df.FareClass == "X")|(coupon_df.FareClass == "Y")]
