#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import bias_mitigation_analysis
ticket=pd.read_csv("Origin_and_Destination_Survey_DB1BTicket_2022_1.csv",nrows=1000000)
income=pd.read_csv("median_income_cityID.csv")
race=pd.read_csv("race_id.csv")
market=pd.read_csv("Origin_and_Destination_Survey_DB1BMarket_2022_1.csv",nrows=1000000)


# In[10]:


bias_mitigation_analysis.bias_mitigation(ticket,market, race, income)


# In[ ]:




