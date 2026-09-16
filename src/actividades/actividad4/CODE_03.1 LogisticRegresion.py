# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:37:59 2024

@author: zaratejo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#%% IMPORT THE DATA SET
data = pd.read_csv('glass.data',header=None)
names = ['ID','Refractive_index','Na','Mg','Al','Si','K',
         'Ca','Ba','Fe','Glass_type']
data.columns = names

# Remove the 'ID' and 'Refractive_index' columns
data = data.drop(['ID', 'Refractive_index'], axis=1)

# Split the data into features and target
X = data.drop('Glass_type', axis=1)
y = data['Glass_type']

# Split the data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

#data_test = pd.concat([X_test, y_test, pd.DataFrame(y_pred, columns=['Glass_type_Pred'], index=y_test.index)], axis=1)


#import seaborn as sns
#sns.scatterplot(x='Si', y='Ca', hue='Glass_type', data=data_test, palette='Set1', marker='o')
#sns.scatterplot(x='Si', y='Ca', hue='Glass_type_Pred', data=data_test, palette='Set1', marker='x')
#plt.show()

#%%
# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
plt.matshow(cm)
plt.title('Confusion matrix of the classifier')
plt.colorbar()
plt.show()
