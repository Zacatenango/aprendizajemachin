#%%
# Homework: one network vs. two networks for wine quality (regression) and color (classification)
import time
import numpy, pandas
import matplotlib.pyplot as pyplot
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, roc_auc_score
import keras
from keras import layers

keras.utils.set_random_seed(297974)          # reproducible weights and batches

# Data: 11 chemical features; two targets
wine = fetch_ucirepo(id=186)
data = wine.data.original
X = data[wine.data.features.columns].values
y_quality = data["quality"].values.astype(float)            # regression target
y_color = (data["color"] == "red").astype(int).values       # 1 = red, 0 = white

X_tr, X_te, q_tr, q_te, c_tr, c_te = train_test_split(
    X, y_quality, y_color, test_size=0.3, random_state=297974, stratify=y_color)
scaler = StandardScaler()                                    # fit on train only (no leakage)
X_tr, X_te = scaler.fit_transform(X_tr), scaler.transform(X_te)

EPOCHS, BATCH = 60, 64
fit_args = dict(epochs=EPOCHS, batch_size=BATCH, validation_split=0.2, verbose=0)

#%%
# (a) Two separate Sequential models
def hidden_stack():
    return [layers.Input(shape=(X.shape[1],)),
            layers.Dense(32, activation="relu"), layers.Dense(16, activation="relu")]

reg = keras.Sequential(hidden_stack() + [layers.Dense(1, activation="linear")])
clf = keras.Sequential(hidden_stack() + [layers.Dense(1, activation="sigmoid")])
reg.compile(optimizer="adam", loss="mse")
clf.compile(optimizer="adam", loss="binary_crossentropy")

t0 = time.time()
h_reg = reg.fit(X_tr, q_tr, **fit_args)
h_clf = clf.fit(X_tr, c_tr, **fit_args)
time_sep = time.time() - t0

#%%
# (b) One Functional model: shared hidden layers, two output heads
inp = layers.Input(shape=(X.shape[1],), name="features")
shared = layers.Dense(32, activation="relu", name="shared_1")(inp)
shared = layers.Dense(16, activation="relu", name="shared_2")(shared)
out_q = layers.Dense(1, activation="linear", name="quality")(shared)
out_c = layers.Dense(1, activation="sigmoid", name="color")(shared)
multi = keras.Model(inputs=inp, outputs=[out_q, out_c])
multi.compile(optimizer="adam",
              loss={"quality": "mse", "color": "binary_crossentropy"})

t0 = time.time()
h_multi = multi.fit(X_tr, {"quality": q_tr, "color": c_tr}, **fit_args)
time_multi = time.time() - t0

#%%
# Compare on the test set
q_sep = reg.predict(X_te, verbose=0).ravel()
p_sep = clf.predict(X_te, verbose=0).ravel()
q_multi, p_multi = (o.ravel() for o in multi.predict(X_te, verbose=0))

results = pandas.DataFrame({
    "Approach": ["Two separate models", "One shared model"],
    "Quality R2": [r2_score(q_te, q_sep), r2_score(q_te, q_multi)],
    "Quality MAE": [mean_absolute_error(q_te, q_sep), mean_absolute_error(q_te, q_multi)],
    "Color accuracy": [accuracy_score(c_te, p_sep > 0.5), accuracy_score(c_te, p_multi > 0.5)],
    "Color AUC": [roc_auc_score(c_te, p_sep), roc_auc_score(c_te, p_multi)],
    "Weights": [reg.count_params() + clf.count_params(), multi.count_params()],
    "Train time (s)": [time_sep, time_multi]}).round(3)
print(results.to_string(index=False))

# Training curves (validation loss) for each target
fig, ax = pyplot.subplots(1, 2, figsize=(11, 4))
ax[0].plot(h_reg.history["val_loss"], label="Separate")
ax[0].plot(h_multi.history["val_quality_loss"], label="Shared")
ax[0].set_title("Quality: validation MSE"); ax[0].set_xlabel("Epoch"); ax[0].legend()
ax[1].plot(h_clf.history["val_loss"], label="Separate")
ax[1].plot(h_multi.history["val_color_loss"], label="Shared")
ax[1].set_title("Color: validation cross-entropy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
pyplot.tight_layout(); pyplot.show()

#%%
# Conclusion (write in your own words after reading the table):
# - Which approach scored better on each target, and by how much?
# - The shared model trains ONE network with fewer weights; is the accuracy cost worth it?
# - Limitation: one split, one seed; small differences may be noise.
