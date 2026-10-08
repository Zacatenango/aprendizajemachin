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

