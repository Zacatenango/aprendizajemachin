
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
import scipy.optimize as opt

import matplotlib.pyplot as plt

# Generate dummy data

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree



#%% Creating the dataset 
X,Y = make_blobs(n_samples=200,
                 centers=[[0],[3]],
                 cluster_std=1.5,
                 n_features=5)

plt.scatter(X,Y,c=Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


#%%

X_tree= X
Y_tree = Y

Classi_1 = DecisionTreeClassifier(max_depth=3, criterion='gini', 
                                  splitter='best' ,min_samples_split =2)
#Classi_1 = DecisionTreeClassifier(max_depth=None, criterion='entropy', splitter='best')
Classi_1.fit(X_tree, Y_tree)
# Predict
y_1 = Classi_1.predict(X_tree)

feature_names =['X1']
target_names =list(['Not-active','Active'])

text_representation = tree.export_text(Classi_1, feature_names=feature_names )
print(text_representation)

fig = plt.figure(figsize=(25,20))
tree.plot_tree(Classi_1,feature_names=feature_names, class_names=target_names,  filled=True)
fig.savefig("Class.pdf")

#%%
X,Y = make_blobs(n_samples=200,
                 centers=[[0,0],[3,3]],
                 cluster_std=1.5,
                 n_features=2)

plt.scatter(X[:,0],X[:,1],c=Y)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()


#from sklearn import preprocessing
#X = preprocessing.scale(X)
#Y = preprocessing.scale(Y)


X_tree= X
Y_tree = Y

#Classi_1 = DecisionTreeClassifier(max_depth=3, criterion='gini', splitter='best' ,min_samples_split =2)
Classi_1 = DecisionTreeClassifier( criterion='gini', splitter='best', 
                                  min_samples_split =10)
Classi_1.fit(X_tree, Y_tree)
# Predict
y_1 = Classi_1.predict(X_tree)

feature_names =['X1', 'X2']
target_names =list(['Not-active','Active'])

text_representation = tree.export_text(Classi_1, feature_names=feature_names )
print(text_representation)

fig = plt.figure(figsize=(25,20))
tree.plot_tree(Classi_1,feature_names=feature_names, class_names=target_names,  filled=True)
fig.savefig("Class2.pdf")