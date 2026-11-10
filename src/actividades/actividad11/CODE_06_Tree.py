# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

#%% Data generation
rng = np.random.RandomState(0)
X = 5 * rng.rand(100, 1) - 2.5
Y = np.ravel(3 * X + 2)
Yrnd = Y + 3 * (0.5 - rng.rand(X.shape[0])) - 10
Yrnd[::20] += 50 * (0.5 - rng.rand(X.shape[0] // 20))

X_plot = np.linspace(-5, 5, 400)[:, None]

plt.figure(figsize=(8, 8))
plt.scatter(X, Yrnd, c='r', label='data-noise')
plt.xlabel('x_p1', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.show()

#%% Build and fit of the decision tree
model = DecisionTreeRegressor(
    splitter='best',
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1
)
model = model.fit(X, Yrnd)
Y_plot = model.predict(X_plot)



plt.figure(figsize=(8, 8))
plt.scatter(X, Yrnd, c='r', label='data-noise')
plt.plot(X_plot, Y_plot, c='g', label='estimation')
plt.xlabel('x_p1', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.legend()
plt.grid()
plt.show()

#%% Decision tree results
indx_leaf = model.apply(X_plot)
importance_predictor = model.feature_importances_
R2 = model.score(X, Yrnd)
depth = model.get_depth()
mse = mean_squared_error(Yrnd, model.predict(X))
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R2 Score: {R2:.2f}')
print(f'Model Depth: {depth}')


#%% Decision tree results
indx_leaf = model.apply(X_plot)
importance_predictor = model.feature_importances_
R2 = model.score(X,Yrnd)
depth = model.get_depth()

#%% View the decision tree
from sklearn import tree
fig = plt.figure(figsize=(25,20))
tree.plot_tree(model) 
#tree.plot_tree(Classi_1,feature_names=feature_names, class_names=target_names,  filled=True)
fig.savefig("Tree.pdf")

#%% Export the decision tree
#https://graphviz.org/download/
#import graphviz
#dot_data = tree.export_graphviz(model, out_file="Tree.pdf")
#graph = graphviz.Source(dot_data)
#graph.render("Tree")

# #%% Save decision tree
# import pickle
# pickle.dump(model,open('model.sav','wb'))

# # Read the decision tree saved
# model = pickle.load(open('model.sav','rb'))