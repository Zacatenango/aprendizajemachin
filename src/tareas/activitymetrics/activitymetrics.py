#%%
# Assignment setup
from ucimlrepo import fetch_ucirepo 
import pandas
import numpy
import matplotlib.pyplot as pyplot
  
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
# Preparation
# We start out by assessing the percentage of 1's we have in y["quality"]
# Since we binarized that column in the assignment setup, we just need to average everything
share_of_ones = y["quality"].mean()
print(f"Share of 1's: {share_of_ones}")

# Use train 70% for train, 30% for test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y["quality"], test_size=0.3, random_state=297974)

# We now scale our data to avoid complaints about convergence
# We only use fit_transform() on training to avoid data leakage
# Reminder to scale the independent variables, not the dependent one!
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
# 1) Run Logistics regression
from sklearn.linear_model import LogisticRegression
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_scaled, Y_train)

#%%
# 2) Calculate Accuracy, Precision, Recall, F1 score (function from class)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

def eval_perform(Y,Yhat):
    accu = accuracy_score(Y,Yhat)
    prec = precision_score(Y,Yhat,average='weighted')
    reca = recall_score(Y,Yhat,average='weighted')
    f1_train = f1_score(Y,Yhat)
    print('\n\t Accu \t Prec \t Reca \t F1\nEval \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu, prec, reca, f1_train))

def evaluate_model(model: LogisticRegression, X_test, Y_test, name):
   Yhat_test = model.predict(X_test)
   eval_perform(Y_test, Yhat_test)
   probabilities_class_1 = model.predict_proba(X_test)[:,1]
   FPR, TPR, thresholds = roc_curve(Y_test, probabilities_class_1)
   integral = auc(FPR, TPR)
   dist = numpy.sqrt(FPR**2 + (1-TPR)**2)
   index = numpy.argmin(dist)
   best_threshold = thresholds[index]
   print(f"Best threshold: {best_threshold}")

   return { "threshold": best_threshold }

# We predict, and evaluate our model.
# For this run, the results are 0.775 accuracy, 0.751 precision, 0.755 recall, and 0.813 F1.
# Recall is the same as accuracy because a recall on weighted average is mathematically identical 
# to accuracy. 
Yhat_test = logistic_regression_model.predict(X_test_scaled)
eval_perform(Y_test, Yhat_test)
evaluate_model(logistic_regression_model, X_test, Y_test, "")

#%%
# 3) Find optimal decision threshold (plot)

#%%
# 4) Plot ROC curve

#%%
# 5) Try a polynomial regression (you choose the grade), repeat steps 2-4 with this model.

#%%
# 6) Conclusion, find the most robust model using AUC.
