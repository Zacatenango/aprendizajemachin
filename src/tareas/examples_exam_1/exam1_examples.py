#%%
# Question 1: Use the following base code to determine the two users that are most similar 
# according to Jaccard's similarity index. Identify the two users with highest Jaccard similarity
import pandas
import numpy

jaccard_index_labels = [ f"User_{X}" for X in range(1,7) ]
jaccard_data = \
{
   "Course_1":  [1, 1, 0, 1, 0, 1],
   'Course_2':  [1, 1, 1, 0, 0, 1], 
   'Course_3':  [0, 0, 1, 1, 0, 1], 
   'Course_4':  [1, 1, 0, 1, 1, 0], 
   'Course_5':  [0, 0, 1, 0, 1, 1], 
   'Course_6':  [1, 1, 1, 0, 0, 0], 
   'Course_7':  [0, 1, 0, 1, 1, 0], 
   'Course_8':  [1, 1, 0, 0, 1, 1], 
   'Course_9':  [0, 0, 1, 1, 0, 1], 
   'Course_10': [1, 1, 0, 1, 0, 0] 
}
jaccard_DF = pandas.DataFrame(jaccard_data, index=jaccard_index_labels)

# Preparation:
# The Jaccard index is "len(both have it) / (len(both have it) + len(only one has it))"
# In other words: "(both are 1) / (at least one is 1)" -- I've used it in patent prosecution to try 
# and convince an IMPI prosecutor that a product was innovative. 
# In mathematical terms: J(A,B) = len(A inner join B) / len(A full join B).
# For binary vectors, the formula simplifies into J(A, B) = M11 / (M11 + M01 + M10) where Mxy means
# "how many positions are worth x in vector A and worth y in vector B".
# We will start out with just two fixed arrays, C and D
C = numpy.array([1, 0, 1, 0, 0, 1])
D = numpy.array([1, 1, 1, 0, 1, 0])
both_1 = (C & D).sum()
at_least_one_1 = (C | D).sum()
jaccard_CD = both_1 / at_least_one_1
jaccard_CD

#%%
# Answer:
# Now we expand to all our users
from itertools import combinations
jaccard_dict = {}
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
# Question 2: Using a Support Vector Machine for classification, determine which of the following 
# kernels provides the highest accuracy score: linear, polynomial, RBF, or sigmoid. Use the 
# following code to load the dataset: 
# Loading data 
from sklearn import datasets 
wine = datasets.load_wine() 
X = wine.data 
Y = wine.target 
Y_st = wine.target_names.tolist() 
X_st = wine.feature_names 

# Use 100% of the data; do not perform a train/test split.

#%%
# Answer:
# Identify the kernel that produces the highest accuracy score.
# Important: Use the default parameters of sklearn.svm.SVC, except for the kernel parameter. Do not standardize the variables. 
from sklearn.svm import SVC
SVM_linear = SVC(kernel="linear")
SVM_poly = SVC(kernel="poly")
SVM_RBF = SVC(kernel="rbf")
SVM_sigmoid = SVC(kernel="sigmoid")

SVM_linear.fit(X,Y)
SVM_poly.fit(X,Y)
SVM_RBF.fit(X,Y)
SVM_sigmoid.fit(X,Y)

# We print the accuracies. Linear is the winner. This is because the rest of the kernels work with
# distances or dot products between raw feature values, which are scale-sensitive operations; the
# <proline> feature is large, around hundreds or thousands, so it drowns out other features.
# Standardization might close the gap between the kernels.
print(f"Linear kernel acc (winner!): {SVM_linear.score(X,Y)}")
print(f"Polynomial kernel acc: {SVM_poly.score(X,Y)}")
print(f"RBF kernel acc: {SVM_RBF.score(X,Y)}")
print(f"Sigmoid kernel acc: {SVM_sigmoid.score(X,Y)}")


#%%
# Question 3: Using the following code as a baseline, how many outliers do you detect in X1 using
# the interquartile range method with a threshold of 1.5?
# Create a DataFrame with the provided data (RI = rango intercuartílico, IQR in Spanish)
data_RI = \
{ 
   'X1': \
   [ 
      -0.82, 0.14, 0.67, -1.05, 1.22, 
       0.31, -0.44, 0.88, -0.19, 1.04, 
      -0.73, 0.55, 0.02, -1.31, 0.76, 
       0.43, -0.58, 1.15, -0.27, 0.95, 
       6.40, -5.80, 7.10, -6.50, 0.24 
   ] 
} 
 

df_RI = pandas.DataFrame(data_RI) 
print(df_RI) 

# Preparation:
# Quantiles in PANDAS are obtained with <dataframe column>.quantile(<number of quantile from 0 to 1>)
# Quantiles are defined as the percentile position in the data where X% of the values are below it.
# The median is quantile 50; quartiles 1 and 3 are quantiles 25 and 75. Quantiles are usually
# expressed with integer figures, but they're not necessarily discrete; continuous quantiles exist
# too, and can be calculated via linear interpolation, e.g. quantile pi.
df_RI_Q3 = df_RI["X1"].quantile(0.75)

#%%
# Answer:
# Calculate the first quartile, Q1
df_RI_Q1 = df_RI["X1"].quantile(0.25)

# Calculate the third quartile, Q3
df_RI_Q3 = df_RI["X1"].quantile(0.75)

# Calculate the interquartile range, IQR
RI = df_RI_Q3 - df_RI_Q1

# Calculate the lower and upper outlier boundaries using a threshold of 1.5
lower_boundary = df_RI_Q1 - (RI * 1.5)
upper_boundary = df_RI_Q3 + (RI * 1.5)

# Report the total number of outliers in X1
outliers = ( (df_RI['X1'] < lower_boundary) | (df_RI['X1'] > upper_boundary) ).sum()
print(f"Quartile 1: {df_RI_Q1}")
print(f"Quartile 3: {df_RI_Q3}")
print(f"Interquartile range: {RI}")
print(f"Outlier boundaries: [{lower_boundary}, {upper_boundary}]")
print(f"{outliers} outliers")

#%%
# Question 4:
# Using the following code as a baseline, implement a Principal Component Regression (PCR). Apply 
# PCA with n_components=2, use the two principal components to predict Y, and print the resulting 
# R^2 score. 
# Create a DataFrame with the provided data 

data_PCA = \
{ 
   'X1': \
   [ 
      -0.628064, 0.454107, -2.408356, 1.873823, 0.677415, 
      -0.354787, -0.432426, 0.421368, -0.063577, -0.382207, 
      0.667001, 0.417055, -0.270609, -0.388792, 0.010513, 
      -0.478169, -0.603163, 0.709334, -0.399958, -1.615381, 
      -0.920891, 0.648195, 0.172168, -0.295073, 0.994199 
   ], 
   'X2': \
   [ 

      2.221172, -0.833474, 2.296370, -1.927899, -0.144170, 
      -1.023506, 1.507433, 0.682005, 1.271547, -0.772579, 
      0.324521, -0.776366, -0.425321, 0.549275, 0.796260, 
      0.511429, -0.426112, -1.099947, 1.509558, 1.281159, 
      0.958367, 1.855177, -0.699722, 2.633824, 2.829990 
   ], 
   'X3': \
   [ 
      0.885491, -0.182709, -0.672890, 0.404781, 0.566716, 
      -1.055556, 0.652896, 0.869389, 0.775485, -0.883938, 
      0.907432, -0.101989, -0.516714, 0.071529, 0.605496, 
      -0.154131, -0.851588, -0.100897, 0.724102, -0.658331, 
      -0.139131, 1.948926, -0.418749, 1.585189, 2.887101 
   ], 
   'Y': \
   [ 
      -4.377992, 1.546347, -6.856543, 5.661865, 1.288684, 
      1.441821, -2.923592, -0.387835, -2.099030, 0.805580, 
      0.409962, 1.780960, 0.411427, -1.221638, -0.588051, 
      -1.615428, -0.815173, 1.894955, -2.796798, -4.226549, 
      -1.489206, -1.549154, 0.649229, -4.327162, -3.173072 
   ] 
} 

df_PCA = pandas.DataFrame(data_PCA) 
print(df_PCA.head()) 

 
#%%
# Answer:
# Standardize the independent variables using StandardScaler
# Important: Do not perform a train/test split. This requirement was put in place because, one day,
# the professor found a very unusual scenario where the same seed number returned different results.
# There are several reasons why this could happen: maybe a different Python version, or a Python
# platform that isn't the standard reference CPython like Pypi or IronPython, or maybe different
# versions of Numpy or scikit-learn...
# Whatever the case, this thing that should never happen, happened; and since the goal is just 
# comparing accuracy metrics between models, the professor told us in class that he determined the 
# train/test split here is not necessary.
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(df_PCA[['X1','X2','X3']])

# Apply PCA with n_components=2
from sklearn.decomposition import PCA
Z = PCA(n_components=2).fit_transform(X_std)

# Fit a linear regression model using the two principal components
from sklearn.linear_model import LinearRegression
linear_regression_model_PCA = LinearRegression()
linear_regression_model_PCA.fit(Z, df_PCA["Y"])

# Print the R2 score
print(f"R² score: {linear_regression_model_PCA.score(Z, df_PCA['Y'])}")


