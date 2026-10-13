#%%
# Case study: Optimal needle replacement policy
# Acoyani Garrido Sandoval, Israel Rivera Guzmán, Margarita Flores Ruiz, Oscar Gutiérrez Godoy, 
# Emilio Valerio Padilla
# ITESO - Master in Data Science - Predictive Modeling - Fall 2026

# Business context:
# A textile company operates 10 machines. Each machine uses one active needle. The machines operate 
# six days per week, between one and three 7.5-hour shifts per day, producing 45, 90, or 135 
# operating hours per week. A needle may fail in service or be replaced before failure.
# Costs:
# Needle replacement: $10,000 for every replacement
# Failure productivity loss: $20,000 when failure occurs
# Total cost of a failure event: $30,000

# Student objective:
# - Clean and validate the raw dataset
# - Develop regressive and classification predictions as needed
# - Train, test and cross-validation are expected
# - Translate predictions into replacement policies
# - Recommend a policy after cost and sensitivity analysis

# Dataset:
# StudyCase_RawData.xlsx contains one week worth of machine needle work observations. Needle_ID
# changes after replacement, Machine_ID remains fixed. The data includes cumulative hours,
# operating intensity, event indicators, and multivariable operating conditions.
#
# Variables and roles
# - Temperature_C - Weekly operating temperature
# - Humidity_Pct - Relative humidity
# - Season - Spring, Summer, Autumn or Winter
# - Fabric_Thickness_mm - Material loading proxy
# - Machine_Speed_RPM - Operating intensity
# - Vibration_mm_s - Mechanical condition indicator
# - Failure - Binary event target
# - Cumulative_Hours_End - Usage and potential lifetime target

#%%
# Assignment setup
import pandas
import numpy

# We 
DF_from_excel = pandas.read_excel("StudyCase_RawData.xlsx", sheet_name="Raw_Data")
audit = []





#%%
# Step 1: data cleaning
# We start out by dropping our duplicates
nr_duplicates = DF_from_excel.duplicated().sum()
DF = DF_from_excel.drop_duplicates().copy()
audi
