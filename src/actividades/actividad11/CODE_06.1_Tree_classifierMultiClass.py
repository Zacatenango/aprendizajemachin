# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import r2_score




#%% Loading data
from sklearn.datasets import load_digits
digits = load_digits()


#%% Selecting data for training and testing
X = digits.data
Y = digits.target

# Spliting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

X_tree= X
Y_tree = Y

#Classi_1 = DecisionTreeClassifier(max_depth=3, criterion='gini', splitter='best' ,min_samples_split =2)
Classi_1 = DecisionTreeClassifier( max_depth=None, criterion='gini', splitter='best')
Classi_1.fit(X_tree, Y_tree)
# Predict
Yhat = Classi_1.predict(X_tree)

#feature_names =['X1', 'X2']
target_names =list(['0','1','2','3','4','5','6','7','8','9'])

text_representation = tree.export_text(Classi_1)#, feature_names=feature_names )
print(text_representation)

fig = plt.figure(figsize=(25,20))
#tree.plot_tree(Classi_1,feature_names=feature_names, class_names=target_names,  filled=True)
tree.plot_tree(Classi_1, class_names=target_names,  filled=True)
fig.savefig("ClassNumber.pdf")




#%% More metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score)

accu = accuracy_score(Y,Yhat)
prec = precision_score(Y,Yhat,average='weighted')
reca = recall_score(Y,Yhat,average='weighted')


print(' \t\t Accu \t Prec \t Reca\n Value \t %0.3f \t %0.3f \t %0.3f\n'%(accu,prec,reca))
