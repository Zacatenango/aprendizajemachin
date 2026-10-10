#%%
# Assignment setup
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 

y["quality"] = (y["quality"] >= 6).astype(int)

#%%
# Use train 70% for train, 30% for test.

#%%
# 1) Run Logistics regression

#%%
# 2) Calculate Accuracy, Precision, Recall, F1 score (function from class)

#%%
# 3) Find optimal decision threshold (plot)

#%%
# 4) Plot ROC curve

#%%
# 5) Try a polynomial regression (you choose the grade), repeat steps 2-4 with this model.

#%%
# 6) Conclusion, find the most robust model using AUC.
