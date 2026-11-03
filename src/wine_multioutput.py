#%%
# Neural Network Activity: wine quality (regression) and wine color (classification) with Keras
#
# Step 1: Library exploration
# ---------------------------------------------------------------------------------------------
# keras.layers.Dense
#   A fully connected layer: output = activation(inputs @ W + b). Every unit is connected to every
#   input. Its main arguments are `units` (number of neurons) and `activation` ('linear', 'relu',
#   'tanh', 'sigmoid', ...). It is the basic building block of both models below.
#
# keras.layers.Input
#   Not a real layer with weights: it declares the shape of the data that enters the network
#   (here shape=(11,) because there are 11 chemical features). In the Functional API it returns
#   a symbolic tensor that the next layers are called on: x = Dense(5)(input_layer).
#   In a Sequential model it is placed as the first element only to fix the input shape.
#
# keras.models.Sequential
#   A container that stacks layers in a single straight line: the output of layer i is the input
#   of layer i+1. It is the simplest way to build a model, but it only allows ONE input and ONE
#   output, with no branches or shared layers.
#
# Differences / when to use each
#   - Sequential: list of layers, one input -> one output. Good for the two separate models (Step 3).
#   - Input + Dense with keras.models.Model (Functional API): layers are connected like functions,
#     so the graph can branch. This allows several inputs, several outputs, and shared layers,
#     which is what Step 6 needs (one shared body, two heads: quality and color).
#   - Dense is used in both styles; Input is required in the Functional API and optional
#     (but recommended) in Sequential.
# ---------------------------------------------------------------------------------------------
import numpy, pandas
import matplotlib.pyplot as pyplot
import tensorflow as tf
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import keras
from keras.layers import Input, Dense
from keras.models import Sequential, Model

keras.utils.set_random_seed(42)          # reproducible weights and batches

#%%
# Step 2: Data preparation and basic analysis
wine_quality = fetch_ucirepo(id=186)
data = wine_quality.data.original
data['color'] = data.color.replace("red", 1).replace("white", 0).astype(int)

X = wine_quality.data.features
Y = data[['quality', 'color']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# The features have very different scales (e.g. total sulfur dioxide ~100s vs. density ~1),
# so they are standardized. The scaler is fit on the training set only to avoid leakage.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Y_train_quality, Y_train_color = Y_train['quality'].values.astype(float), Y_train['color'].values
Y_test_quality, Y_test_color = Y_test['quality'].values.astype(float), Y_test['color'].values

print(data.shape)
print(data.describe().T.round(2))

# Visual analysis
fig, ax = pyplot.subplots(1, 3, figsize=(15, 4))
data['quality'].value_counts().sort_index().plot.bar(ax=ax[0], color="tab:blue")
ax[0].set_title("Quality distribution"); ax[0].set_xlabel("Quality"); ax[0].set_ylabel("Count")
data['color'].map({1: "red", 0: "white"}).value_counts().plot.bar(ax=ax[1], color=["silver", "darkred"])
ax[1].set_title("Color distribution (imbalanced: ~75% white)"); ax[1].set_ylabel("Count")
pandas.crosstab(data['quality'], data['color'].map({1: "red", 0: "white"})).plot.bar(ax=ax[2])
ax[2].set_title("Quality by color"); ax[2].set_xlabel("Quality")
pyplot.tight_layout(); pyplot.show()

corr = data.drop(columns=['is_red'], errors='ignore').corr(numeric_only=True)
fig, ax = pyplot.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im); ax.set_title("Correlation matrix"); pyplot.tight_layout(); pyplot.show()

fig, axes = pyplot.subplots(3, 4, figsize=(15, 9))
for a, col in zip(axes.ravel(), X.columns):
    data.boxplot(column=col, by='color', ax=a)
    a.set_title(col); a.set_xlabel("color (0=white, 1=red)")
axes.ravel()[-1].axis("off")
pyplot.suptitle("Features by color"); pyplot.tight_layout(); pyplot.show()
# Observations: quality is concentrated in 5-6 (hard regression target, weakly correlated with
# features; alcohol has the strongest correlation). Color is clearly separable by features such
# as total sulfur dioxide, volatile acidity and chlorides, so classification should be easy.

#%%
# Step 3: Two separate Sequential models, several configurations
# Note: the template uses shape=(1,), but X has 11 features, so the input shape is (11,).
N_FEATURES = X_train.shape[1]
SEARCH_EPOCHS, BATCH = 50, 32

def build_sequential(hidden, output_activation, name):
    """hidden: list of (units, activation) for the hidden layers."""
    model_layers = [tf.keras.Input(shape=(N_FEATURES,), name='input_layer')]
    for k, (units, act) in enumerate(hidden, start=1):
        model_layers.append(Dense(units, activation=act, name=f'hidden_layer{k}'))
    model_layers.append(Dense(1, activation=output_activation, name='output_layer'))
    return Sequential(model_layers, name=name)

configs = {
    "statement: 2 linear, 5 tanh": [(2, 'linear'), (5, 'tanh')],
    "16 tanh, 8 tanh":             [(16, 'tanh'), (8, 'tanh')],
    "32 relu, 16 relu":            [(32, 'relu'), (16, 'relu')],
    "64 relu, 32 relu, 16 relu":   [(64, 'relu'), (32, 'relu'), (16, 'relu')],
}

# Model selection uses a validation split of the TRAINING set; the test set is kept for Step 5.
rows = []
for cname, hidden in configs.items():
    m1 = build_sequential(hidden, 'linear', 'Sequential_regression')
    m1.compile(optimizer='adam', loss='mse', metrics=['mse'])
    h1 = m1.fit(X_train, Y_train_quality, epochs=SEARCH_EPOCHS, batch_size=BATCH,
                validation_split=0.2, verbose=0)
    m2 = build_sequential(hidden, 'sigmoid', 'Sequential_classification')
    m2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h2 = m2.fit(X_train, Y_train_color, epochs=SEARCH_EPOCHS, batch_size=BATCH,
                validation_split=0.2, verbose=0)
    rows.append({"Configuration": cname,
                 "Quality val MSE": min(h1.history['val_mse']),
                 "Color val accuracy": max(h2.history['val_accuracy']),
                 "Params (reg)": m1.count_params()})
config_results = pandas.DataFrame(rows).round(4)
print(config_results.to_string(index=False))

best_reg_config = config_results.loc[config_results["Quality val MSE"].idxmin(), "Configuration"]
best_clf_config = config_results.loc[config_results["Color val accuracy"].idxmax(), "Configuration"]
print("Best regression config:", best_reg_config)
print("Best classification config:", best_clf_config)

#%%
# Step 4: Optimizer selection (using the best configuration of each model)
optimizers = {
    "SGD lr=0.01":            lambda: keras.optimizers.SGD(learning_rate=0.01),
    "SGD lr=0.01 momentum=0.9": lambda: keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSprop lr=0.001":       lambda: keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam lr=0.001":          lambda: keras.optimizers.Adam(learning_rate=0.001),
    "Adam lr=0.01":           lambda: keras.optimizers.Adam(learning_rate=0.01),
}

rows, opt_histories = [], {}
for oname, make_opt in optimizers.items():
    m1 = build_sequential(configs[best_reg_config], 'linear', 'Sequential_regression')
    m1.compile(optimizer=make_opt(), loss='mse', metrics=['mse'])
    h1 = m1.fit(X_train, Y_train_quality, epochs=SEARCH_EPOCHS, batch_size=BATCH,
                validation_split=0.2, verbose=0)
    m2 = build_sequential(configs[best_clf_config], 'sigmoid', 'Sequential_classification')
    m2.compile(optimizer=make_opt(), loss='binary_crossentropy', metrics=['accuracy'])
    h2 = m2.fit(X_train, Y_train_color, epochs=SEARCH_EPOCHS, batch_size=BATCH,
                validation_split=0.2, verbose=0)
    opt_histories[oname] = (h1, h2)
    rows.append({"Optimizer": oname,
                 "Quality val MSE": min(h1.history['val_mse']),
                 "Color val accuracy": max(h2.history['val_accuracy'])})
opt_results = pandas.DataFrame(rows).round(4)
print(opt_results.to_string(index=False))

fig, ax = pyplot.subplots(1, 2, figsize=(12, 4))
for oname, (h1, h2) in opt_histories.items():
    ax[0].plot(h1.history['val_mse'], label=oname)
    ax[1].plot(h2.history['val_loss'], label=oname)
ax[0].set_title("Regression: validation MSE"); ax[0].set_xlabel("Epoch"); ax[0].set_ylim(0, 2)
ax[1].set_title("Classification: validation cross-entropy"); ax[1].set_xlabel("Epoch")
ax[0].legend(); pyplot.tight_layout(); pyplot.show()

best_reg_opt = opt_results.loc[opt_results["Quality val MSE"].idxmin(), "Optimizer"]
best_clf_opt = opt_results.loc[opt_results["Color val accuracy"].idxmax(), "Optimizer"]
print("Best regression optimizer:", best_reg_opt)
print("Best classification optimizer:", best_clf_opt)
# Justification: the final optimizer for each model is the one with the best validation score
# (lowest MSE / highest accuracy) in the table above. With seed 42 this was SGD + momentum for
# quality (momentum smooths the noisy gradient of a target that is mostly 5-6) and RMSprop for
# color. The differences are small (~0.03 MSE, ~0.1% accuracy), so for color any optimizer works;
# the curves show adaptive optimizers (Adam/RMSprop) reach a low loss in the first few epochs,
# while a high learning rate (Adam lr=0.01) is faster but its validation curve is noisier.

#%%
# Step 5: Train and evaluate the final separate models on the test set
FINAL_EPOCHS = 100

model1 = build_sequential(configs[best_reg_config], 'linear', 'Sequential_regression')
model1.compile(optimizer=optimizers[best_reg_opt](), loss='mse', metrics=['mse'])
h_model1 = model1.fit(X_train, Y_train_quality, epochs=FINAL_EPOCHS, batch_size=BATCH,
                      validation_split=0.2, verbose=0)

model2 = build_sequential(configs[best_clf_config], 'sigmoid', 'Sequential_classification')
model2.compile(optimizer=optimizers[best_clf_opt](), loss='binary_crossentropy', metrics=['accuracy'])
h_model2 = model2.fit(X_train, Y_train_color, epochs=FINAL_EPOCHS, batch_size=BATCH,
                      validation_split=0.2, verbose=0)

model1.summary(); model2.summary()

q_sep = model1.predict(X_test, verbose=0).ravel()
p_sep = model2.predict(X_test, verbose=0).ravel()
mse_sep = mean_squared_error(Y_test_quality, q_sep)
acc_sep = accuracy_score(Y_test_color, (p_sep > 0.5).astype(int))
print(f"Regression model (quality)   test MSE:      {mse_sep:.4f}")
print(f"Classification model (color) test accuracy: {acc_sep:.4f}")
# Reference: always predicting the mean quality gives MSE = variance of quality
print(f"Baseline MSE (predict mean): {numpy.var(Y_test_quality):.4f}")

fig, ax = pyplot.subplots(1, 2, figsize=(12, 4))
ax[0].plot(h_model1.history['loss'], label="train"); ax[0].plot(h_model1.history['val_loss'], label="validation")
ax[0].set_title("model1 (quality): MSE"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(h_model2.history['accuracy'], label="train"); ax[1].plot(h_model2.history['val_accuracy'], label="validation")
ax[1].set_title("model2 (color): accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
pyplot.tight_layout(); pyplot.show()

#%%
# Step 6: Combined model with multiple outputs (Functional API: Input + Dense + Model)
opt = optimizers[best_reg_opt]()      # same optimizer chosen in Step 4 for regression

# Input layer
input_layer = Input(shape=(11,), name='input_layer')

# Shared hidden layers
shared_layer = Dense(5, activation='sigmoid')(input_layer)
shared_layer = Dense(10, activation='tanh')(shared_layer)
shared_layer = Dense(10, activation='tanh')(shared_layer)

# Output for quality (regression)
output_quality = Dense(1, activation='linear', name='quality_output')(shared_layer)

# Output for color (classification)
output_color = Dense(1, activation='sigmoid', name='color_output')(shared_layer)

# Create model
model = Model(inputs=input_layer, outputs=[output_quality, output_color])

model.compile(optimizer=opt,
              loss={'quality_output': 'mse', 'color_output': 'binary_crossentropy'},
              metrics={'quality_output': 'mae', 'color_output': 'accuracy'})
model.summary()

# Train the model
history = model.fit(X_train,
                    {'quality_output': Y_train_quality, 'color_output': Y_train_color},
                    epochs=100, batch_size=32, verbose=0,
                    validation_data=(X_test, {'quality_output': Y_test_quality, 'color_output': Y_test_color}))

#%%
# Step 7: Evaluate the combined model and compare with the separate models
q_multi, p_multi = (o.ravel() for o in model.predict(X_test, verbose=0))
mse_multi = mean_squared_error(Y_test_quality, q_multi)
acc_multi = accuracy_score(Y_test_color, (p_multi > 0.5).astype(int))
print(f"Combined model, quality test MSE:     {mse_multi:.4f}")
print(f"Combined model, color test accuracy:  {acc_multi:.4f}")

results = pandas.DataFrame({
    "Approach": ["Two separate Sequential models", "One combined Functional model"],
    "Quality MSE (test)": [mse_sep, mse_multi],
    "Color accuracy (test)": [acc_sep, acc_multi],
    "Weights": [model1.count_params() + model2.count_params(), model.count_params()]}).round(4)
print(results.to_string(index=False))

fig, ax = pyplot.subplots(1, 3, figsize=(16, 4))
ax[0].plot(history.history['quality_output_loss'], label="train")
ax[0].plot(history.history['val_quality_output_loss'], label="test")
ax[0].set_title("Combined model: quality MSE"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(history.history['color_output_accuracy'], label="train")
ax[1].plot(history.history['val_color_output_accuracy'], label="test")
ax[1].set_title("Combined model: color accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
ax[2].scatter(Y_test_quality + numpy.random.uniform(-0.2, 0.2, len(Y_test_quality)), q_multi, s=4, alpha=0.4, label="combined")
ax[2].scatter(Y_test_quality + numpy.random.uniform(-0.2, 0.2, len(Y_test_quality)), q_sep, s=4, alpha=0.4, label="separate")
ax[2].plot([3, 9], [3, 9], "k--"); ax[2].set_xlabel("True quality (jittered)"); ax[2].set_ylabel("Predicted")
ax[2].set_title("Quality: predicted vs. true"); ax[2].legend()
pyplot.tight_layout(); pyplot.show()

#%%
# Step 8: Conclusion
# - Classification: The separate classification model had a very slightly better accuracy than the
#   shared model.
# - Regression: The separate regression model had a slightly better MSE than the shared model.
# - Conclusion: The separate models were only slightly better than the shared model, with
#   approximately 7 times the amount of weights. The combined model was almost just as good, and
#   was much more efficient in resource usage.
