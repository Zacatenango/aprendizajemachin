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
# Since we binarized that column in the assignment setup, we just need to average everything. This
# 
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
   sensitivity_best_thresh = TPR[index]
   specificity_best_thresh = 1 - FPR[index]

   # Print output
   print(f"Best threshold: {best_threshold}")
   print(f"Sensitivity: {sensitivity_best_thresh}")
   print(f"Specificity: {specificity_best_thresh}")

   # Plot specificity and sensitivity with best threshold
   # (this answers question 3)
   pyplot.figure()
   pyplot.plot(thresholds, 1-FPR, label="Specificity")
   pyplot.plot(thresholds, TPR, label="Sensitivity")
   pyplot.title(f"Specificity and sensitivity: {name}")
   pyplot.axvline(best_threshold, color="red")
   pyplot.legend()

   # Plot ROC curve (this answers question 4)
   pyplot.figure()
   pyplot.plot(FPR, TPR, label=f"{name} (AUC = {integral:.3f})")
   pyplot.plot([0,1], [0,1], "--")
   pyplot.xlabel("1 - specificity")
   pyplot.ylabel("Sensitivity")
   pyplot.legend()
   pyplot.title(f"ROC curve: {name}")
   
   return { "threshold": best_threshold, "integral": integral }

# We predict, and evaluate our model.
# For this run, the results are 0.755 accuracy, 0.751 precision, 0.755 recall, and 0.813 F1.
# Recall is the same as accuracy because a recall on weighted average is mathematically identical 
# to accuracy.
logistic_result = evaluate_model(logistic_regression_model, X_test_scaled, Y_test, "Logistic")

#%%
# 5) Try a polynomial regression (you choose the grade), repeat steps 2-4 with this model.
# I chose grade 2 because higher grades imply a higher risk of overfitting
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree=2)
X_train_poly = polynomial.fit_transform(X_train_scaled)
X_test_poly = polynomial.transform(X_test_scaled)

# We create a new model and fit it on our polynomial-transformed independent variables
# We need more iterations because polynomial transformation increased the amount of features
polynomial_regression_model = LogisticRegression(max_iter=5000)
polynomial_regression_model.fit(X_train_poly, Y_train)
polynomial_result = evaluate_model(polynomial_regression_model, X_test_poly, Y_test, "Polynomial")


# %%
# 6) Conclusion, find the most robust model using AUC.
# We use AUC because it measures the model's performance across all thresholds, not just at 0.5
# Compared with the baseline prediction accuracy, which is the share of ones in the independent
# variable, both models have an AUC higher than the baseline, which suggests they are both robust.
# While the polynomial regression's accuracy is just marginally higher by 1.2 percent points, it 
# still is decently more robust because it has a 4.2 percent points higher specificity (less risk of
# mistakenly labeling bad wines as good).
best_model = "polynomial" if polynomial_result["integral"] > logistic_result["integral"] else "logistic"
print(f"The winner model is: {best_model}!")
print(f"AUC of polynomial regression: {polynomial_result['integral']}")
print(f"AUC of logistic regression: {logistic_result['integral']}")
print(f"Baseline (share of ones): {share_of_ones}")

# Limitations:
# - This is only a single 70/30 split, cross-validation would confirm whether the 0.014 increase in
#   polynomial regression's AUC is real or just noise.
# - The best threshold was found on the test set, which leaks test information.
# - The class function mixes weighted precision and recall with binary F1
