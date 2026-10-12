#%% Setup
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from itertools import combinations
import numpy

#%%  Loading data

iris = datasets.load_iris()
X=iris.data
Y = iris.target
Y_st=iris.target_names.tolist()
X_st=iris.feature_names

#USE 100% of DATA (no train/test split)

#%% 
LDAmodel = LinearDiscriminantAnalysis(store_covariance=True, n_components=None)
LDAmodel_fit = LDAmodel.fit(X,Y)
X_new = LDAmodel_fit.transform(X)
Yhat = LDAmodel_fit.predict(X)

accu = accuracy_score(Y,Yhat)
prec = precision_score(Y,Yhat,average='weighted')
reca = recall_score(Y,Yhat,average='weighted')
f1_train = f1_score(Y,Yhat,average="weighted")
print('\n\t Accu \t Prec \t Reca \t F1\nEval \t %0.3f \t %0.3f \t %0.3f \t %0.3f' % (accu, prec, reca, f1_train))

#%%
# Use the following base code to determine the users that are most similar according to the Jaccard 
# similarity index.

# Define the data
data = {
    'Q1': [1, 1, 1, 1, 1, 0],
    'Q2': [1, 1, 0, 0, 1, 0],
    'Q3': [0, 1, 0, 1, 1, 0],
    'Q4': [0, 1, 1, 1, 1, 0],
    'Q5': [1, 1, 1, 0, 1, 0],
    'Q6': [1, 1, 1, 1, 1, 0],
    'Q7': [0, 1, 0, 0, 1, 0],
    'Q8': [1, 1, 0, 1, 1, 0],
    'Q9': [0, 1, 0, 0, 1, 0],
    'Q10': [0, 0, 0, 1, 1, 0]
}

# Define row labels
index_labels = ['A', 'B', 'C', 'D', 'E', 'F']

# Create DataFrame
df = pd.DataFrame(data, index=index_labels)

#%%
# Answer

jaccard_dict = {}
jaccard_DF = df
for A, B in combinations(jaccard_DF.index.values, 2):
   arr_A = numpy.array(jaccard_DF.loc[A].values)
   arr_B = numpy.array(jaccard_DF.loc[B].values)
   both_1 = (arr_A & arr_B).sum()
   at_least_one_1 = (arr_A | arr_B).sum()
   jaccard_dict[f"{A}, {B}"] = both_1 / at_least_one_1

# We sort it by descending value to identify the top 2 most similar users
jaccard_dict_sorted = sorted(jaccard_dict.items(), key=lambda X: X[1], reverse=True)

# This gave us a list of tuples from where we can fetch our top 2 users
print(f"Most similar users: {jaccard_dict_sorted[0][0]}: {jaccard_dict_sorted[0][1]}")
