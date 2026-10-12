#%% Setup
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
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
import numpy
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

#%%
import numpy as np
import pandas as pd

# Create a DataFrame with the provided data
data = {
    'X1': [0.2088635950047554, -0.7198442083947086, -0.6017066122293969, -0.38508228041631654, -0.5622875292409727,
           -0.22577630048653566, 5.132674331133561, -5.182702182736474, 1.5792128155073915, 1.5230298564080254,
           -0.9080240755212109, -0.3092123758512146, 0.5425600435859647, -2.1967188783751195, 9.154021177020741,
           0.19686123586912352, 0.3436182895684614, -7.020530938773524, -0.11564828238824053, -1.0577109289559004,
           0.37569801834567196, 1.030999522495951, 0.4967141530112327, 0.24196227156603412, -0.5443827245251827],
    'X2': [-1.9596701238797756, -0.4606387709597875, 1.8522781845089378, -0.6769220003059587, -1.0128311203344238,
           0.06752820468792384, 0.9707754934804038, -8.084936028931876, 0.7674347291529088, -0.23415337472333597,
           -1.4123037013352915, 0.33126343140356396, -0.46341769281246226, 3.5711257151174642, 3.2875110965968446,
           0.7384665799954104, -1.763040155362734, -3.2766214659776822, -0.3011036955892888, 0.822544912103189,
           -0.600638689918805, 0.9312801191161986, -0.13826430117118466, -1.913280244657798, 0.11092258970986608],
    'X3': [-1.3281860488984305, 1.0571222262189157, -0.013497224737933921, 0.6116762888408679, 0.3142473325952739,
           -1.4247481862134568, 9.686449905328892, -5.017570435845365, -0.4694743859349521, -0.23413695694918055,
           1.465648768921554, 0.9755451271223592, -0.46572975357025687, 14.778940447415161, -5.297602037670388,
           0.1713682811899705, 0.324083969394795, -3.921081531321576, -1.4785219903674274, -1.2208436499710222,
           -0.2916937497932768, -0.8392175232226385, 0.6476885381006925, -1.7249178325130328, -1.1509935774223028],
    'Y': [-0.03582603910995154, 0.8219025043752238, 0.36139560550841393, -0.29900735046586746, 1.356240028570823,
          1.0035328978920242, 0.051134566424608896, 2.9612027706457607, -1.1063349740060282, -0.18565897666381712,
          -0.07201012158033385, -1.9875689146008928, -1.1962066240806708, -14.635149481321186, 2.6105527217988933,
          1.5646436558140062, 0.08704706823817122, -2.345871333751469, -2.6197451040897444, 1.5380365664659692,
          -0.6451197546051243, 0.0917607765355023, -0.47917423784528995, 0.812525822394198, 0.36163602504763415]
}

df = pd.DataFrame(data)

#%%
# Implement PLS with n=2
df_PLS = df
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
PLS = PLSRegression(n_components=2)
X = df_PLS[["X1", "X2", "X3"]]
Y = df_PLS["Y"]

# Show the transformation and perform a linear regression
PLS.fit(X,Y)
transformation = PLS.transform(X)
print(transformation[:5])
linear_reg = LinearRegression().fit(transformation, Y)

# Your code must print the R2 score
Yhat = PLS.predict(X)
print(f"R2 score: {linear_reg.score(transformation, Y)}")
