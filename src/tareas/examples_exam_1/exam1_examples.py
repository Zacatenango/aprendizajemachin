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


#%%
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

