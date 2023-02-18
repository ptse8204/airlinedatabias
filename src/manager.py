
import pandas as pd
import zipfile
from other import gen_ticket_coupon

# This function allow for easy extraction of coupon zip
def load_coupon(path, year, quarter):
  with zipfile.ZipFile(
    "{path_f}/Origin_and_Destination_Survey_DB1BCoupon_{year_f}_{quarter_f}.zip".format(
        path_f = path, year_f = year, quarter_f = quarter),"r") as zip_ref:
    df_path = zip_ref.extract([i for i in zip_ref.namelist() if ".csv" in i][0])
  return df_path # Path of result file

# This function allow for easy access of coupon dataframe
def load_coupon_df(path, year, quarter):
  return pd.read_csv(load_coupon(path, year, quarter))

# This function allow for easy extraction of ticket zip
def load_ticket(path, year, quarter):
  with zipfile.ZipFile(
    "{path_f}/Origin_and_Destination_Survey_DB1BTicket_{year_f}_{quarter_f}.zip".format(
        path_f = path, year_f = year, quarter_f = quarter),"r") as zip_ref:
    df_path = zip_ref.extract([i for i in zip_ref.namelist() if ".csv" in i][0])
  return df_path # Path of result file

# This function allow for easy access of coupon dataframe
def load_ticket_df(path, year, quarter):
  return pd.read_csv(load_ticket(path, year, quarter))

# zip -- boolean --True = is a zip path --False = is a .csv path
# feed func_arg with any variables to succesfully to run your function
def run_result_among_ticket(year_start, quarter_start, year_end, quarter_end, 
                            path, zip, function, *func_arg):
  result_storage = []
  for y in range(year_start, year_end + 1):
    q_loader_s = 1
    q_loader_e = 5
    if y == year_start:
      q_loader_s = quarter_start
    elif y == year_end:
      q_loader_e = quarter_end + 1
    for q in range(q_loader_s, q_loader_e):
      if zip:
        curr_df = load_ticket_df(path, y, q)
      else:
        curr_df = pd.read_csv(
              "{path_f}/Origin_and_Destination_Survey_DB1BTicket_{year_f}_{quarter_f}.{file_type}"
              .format(path_f = path, year_f = y, quarter_f = q, file_type = "zip" if zip else "csv"))
      result_storage.append(function(curr_df, *func_arg))
      del curr_df
  return result_storage

def run_result_among_coupon(year_start, quarter_start, year_end, quarter_end, 
                            path, zip, function, *func_arg):
  result_storage = []
  for y in range(year_start, year_end + 1):
    q_loader_s = 1
    q_loader_e = 5
    if y == year_start:
      q_loader_s = quarter_start
    elif y == year_end:
      q_loader_e = quarter_end + 1
    for q in range(q_loader_s, q_loader_e):
      if zip:
        curr_df = load_coupon_df(path, y, q)
      else:
        curr_df = pd.read_csv(
              "{path_f}/Origin_and_Destination_Survey_DB1BCoupon_{year_f}_{quarter_f}.{file_type}"
              .format(path_f = path, year_f = y, quarter_f = q, file_type = "zip" if zip else "csv"))
      result_storage.append(function(curr_df, *func_arg))
      del curr_df
  return result_storage

# run orginal combined only
def run_result_among_combined(
    year_start, quarter_start, year_end, quarter_end,
    ticket_path, coupon_path, zip, function, *func_arg):
  result_storage = []
  for y in range(year_start, year_end + 1):
    q_loader_s = 1
    q_loader_e = 5
    if y == year_start:
      q_loader_s = quarter_start
    elif y == year_end:
      q_loader_e = quarter_end + 1
    for q in range(q_loader_s, q_loader_e):
      if zip:
        curr_coupon_df = load_coupon_df(coupon_path, y, q)
        curr_ticket_df = load_ticket_df(ticket_path, y, q)
      else:
        curr_coupon_df = pd.read_csv(
              "{path_f}/Origin_and_Destination_Survey_DB1BCoupon_{year_f}_{quarter_f}.{file_type}"
              .format(path_f = coupon_path, year_f = y, quarter_f = q, file_type = "zip" if zip else "csv"))
        curr_ticket_df = pd.read_csv(
              "{path_f}/Origin_and_Destination_Survey_DB1BTicket_{year_f}_{quarter_f}.{file_type}"
              .format(path_f = ticket_path, year_f = y, quarter_f = q, file_type = "zip" if zip else "csv"))
      curr_df = gen_ticket_coupon(curr_ticket_df, curr_coupon_df)
      del curr_ticket_df
      del curr_coupon_df
      result_storage.append(function(curr_df, *func_arg))
      del curr_df
  return result_storage

# run different types of combine func
# combo_type = "default": runs the orginal default function, which
# uses the last coupon as destination (not accurate and not recommend,
#  as >60% of tickets are round-trips)
# combo_type = "segment": create combine with all segment(coupon)
# combo_type = "midpoint": set destination as median coupon
def run_result_among_combined_s(
    year_start, quarter_start, year_end, quarter_end,
    ticket_path, coupon_path, zip, combo_type="default",function, *func_arg):
  result_storage = []
  for y in range(year_start, year_end + 1):
    q_loader_s = 1
    q_loader_e = 5
    if y == year_start:
      q_loader_s = quarter_start
    elif y == year_end:
      q_loader_e = quarter_end + 1
    for q in range(q_loader_s, q_loader_e):
      if zip:
        curr_coupon_df = load_coupon_df(coupon_path, y, q)
        curr_ticket_df = load_ticket_df(ticket_path, y, q)
      else:
        curr_coupon_df = pd.read_csv(
              "{path_f}/Origin_and_Destination_Survey_DB1BCoupon_{year_f}_{quarter_f}.{file_type}"
              .format(path_f = coupon_path, year_f = y, quarter_f = q, file_type = "zip" if zip else "csv"))
        curr_ticket_df = pd.read_csv(
              "{path_f}/Origin_and_Destination_Survey_DB1BTicket_{year_f}_{quarter_f}.{file_type}"
              .format(path_f = ticket_path, year_f = y, quarter_f = q, file_type = "zip" if zip else "csv"))
      if combo_type == "segment"
        curr_df = combined_based_coupon(curr_ticket_df, curr_coupon_df)
      elif combo_type == "midpoint":
        curr_df = gen_ticket_coupon_median(curr_ticket_df, curr_coupon_df)
      else:
        curr_df = gen_ticket_coupon(curr_ticket_df, curr_coupon_df)
      del curr_ticket_df
      del curr_coupon_df
      result_storage.append(function(curr_df, *func_arg))
      del curr_df
  return result_storage
