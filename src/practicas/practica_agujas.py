#%%
# Case study: Optimal needle replacement policy
# Acoyani Garrido Sandoval, Israel Rivera Guzmán, Margarita Flores Ruiz, Oscar Gutiérrez Godoy, 
# Emilio Valerio Padilla
# ITESO - Master in Data Science - Predictive Modeling - Fall 2026

# Business context:
# A textile company operates 10 machines. Each machine uses one active needle. The machines operate 
# six days per week, between one and three 7.5-hour shifts per day, producing 45, 90, or 135 
# operating hours per week. A needle may fail in service or be replaced before failure.
# Costs:
# Needle replacement: $10,000 for every replacement
# Failure productivity loss: $20,000 when failure occurs
# Total cost of a failure event: $30,000

# Student objective:
# - Clean and validate the raw dataset
# - Develop regressive and classification predictions as needed
# - Train, test and cross-validation are expected
# - Translate predictions into replacement policies
# - Recommend a policy after cost and sensitivity analysis

# Dataset:
# StudyCase_RawData.xlsx contains one week worth of machine needle work observations. Needle_ID
# changes after replacement, Machine_ID remains fixed. The data includes cumulative hours,
# operating intensity, event indicators, and multivariable operating conditions.
#
# Variables and roles
# - Temperature_C - Weekly operating temperature
# - Humidity_Pct - Relative humidity
# - Season - Spring, Summer, Autumn or Winter
# - Fabric_Thickness_mm - Material loading proxy
# - Machine_Speed_RPM - Operating intensity
# - Vibration_mm_s - Mechanical condition indicator
# - Failure - Binary event target
# - Cumulative_Hours_End - Usage and potential lifetime target

#%%
# Assignment setup
import pandas
import numpy

# In addition to loading the data, we add an "audit" list. Every time we do something with our
# data, we will append to it an entry that specifies what we did and the result. This will help us
# keep track of what we do.
DF_from_excel = pandas.read_excel("StudyCase_RawData.xlsx", sheet_name="Raw_Data")
audit = []





#%%
# Step 1: data cleaning
# We start out by dropping our duplicates
nr_duplicates = DF_from_excel.duplicated().sum()
DF = DF_from_excel.drop_duplicates().copy()
audit.append({ "Issue": "Dataset had exact duplicate rows", "Rows": nr_duplicates, "Treatment": "Removed" })
print(f"Number of duplicates removed: {nr_duplicates}")

# We verify that each week has 10 machines
DF.groupby("Week").size().value_counts()





#%%
# Step 2: standardize Season labels
before = DF["Season"].value_counts()
DF["Season"] = DF["Season"].str.strip().str.title().replace(
    {"Wnter": "Winter", "Summmer": "Summer", "Sprng": "Spring", "Autum": "Autumn"})
n_fixed = int((DF_from_excel.drop_duplicates()["Season"] != DF["Season"]).sum())
audit.append({"Issue": "Misspelled/miscased Season labels", "Rows": n_fixed,
              "Treatment": "Standardized to Winter/Spring/Summer/Autumn"})
print(DF["Season"].value_counts())   # must show exactly 4 seasons





#%%
# Step 3: outliers -> NaN (physical limits + IQR rule), then median imputation
sensors = ["Temperature_C", "Humidity_Pct", "Fabric_Thickness_mm",
           "Machine_Speed_RPM", "Vibration_mm_s"]
for col in sensors:
    q1, q3 = DF[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 3 * iqr, q3 + 3 * iqr        # 3x IQR: only extreme values
    if col == "Humidity_Pct":
        high = min(high, 100)                     # humidity can't exceed 100%
    mask = (DF[col] < low) | (DF[col] > high)
    audit.append({"Issue": f"{col}: extreme outliers", "Rows": int(mask.sum()),
                  "Treatment": f"Set to NaN (outside [{low:.1f}, {high:.1f}])"})
    DF.loc[mask, col] = numpy.nan

impute_cols = sensors + ["Shifts_Per_Day"]
for col in impute_cols:
    n_na = int(DF[col].isna().sum())
    DF[col] = DF[col].fillna(DF[col].median())
    audit.append({"Issue": f"{col}: missing values (incl. removed outliers)",
                  "Rows": n_na, "Treatment": "Filled with column median"})

audit_table = pandas.DataFrame(audit)
print(audit_table.to_string(index=False))





#%%
# Step 4: EDA. One row per needle: lifetime and how it ended
import matplotlib.pyplot as pyplot

needles = (DF.sort_values("Week")
             .groupby("Needle_ID")
             .agg(Machine=("Machine_ID", "first"),
                  Hours=("Cumulative_Hours_End", "max"),
                  Failed=("Failure", "max"),
                  Preventive=("Replace_Before_Failure", "max")))
needles["Status"] = numpy.select(
    [needles.Failed == 1, needles.Preventive == 1],
    ["Failed", "Preventive"], default="Censored (still running)")
print(needles["Status"].value_counts())
print(needles.loc[needles.Failed == 1, "Hours"].describe())

# Chart 1: hours at failure
pyplot.figure()
pyplot.hist(needles.loc[needles.Failed == 1, "Hours"], bins=12, edgecolor="black")
pyplot.xlabel("Cumulative hours at failure"); pyplot.ylabel("Needles")
pyplot.title("Needle lifetime at failure (n=30)")
pyplot.savefig("eda_failure_hours.png", dpi=150, bbox_inches="tight")

# Chart 2: weekly failure rate by cumulative-hour band
DF["Hour_Band"] = pandas.cut(DF["Cumulative_Hours_End"],
                             bins=range(0, 10500, 1000))
rate = DF.groupby("Hour_Band", observed=True)["Failure"].mean()
print(rate)
pyplot.figure()
rate.plot(kind="bar", edgecolor="black")
pyplot.ylabel("Weekly failure rate"); pyplot.title("Failure risk rises with needle age")
pyplot.savefig("eda_failure_rate_band.png", dpi=150, bbox_inches="tight")

# Failure rate by season
print(DF.groupby("Season")["Failure"].mean())
pyplot.show()

# EDA conclusions:
# - No failures before 3,000 hours. This is a natural floor for any replacement threshold.
# - Risk climbs with age: weekly failure rate about 1% below 5,000 h, 3–4% from 5,000 to 7,000 h, 
#   then 6–14% above 7,000 h. This is a classic wear-out pattern, and hints at a fixed hour policy.
# - Season effect is small (0.8–1.5% weekly, summer highest) and rests on only 30 failures. It
#   should probably be reported as weak.
# - 10 censored needles: their true lifetime is unknown; limitation of the dataset.





#%%
# Step 5: cost frontier. Simulate a fixed-hour replacement policy T on each needle's observed life
def simulate(T):
    cost = hours = failures = replacements = 0
    for H, status in zip(needles.Hours, needles.Status):
        if status == "Failed" and H <= T:          # fails before reaching T
            cost += 30_000; hours += H; failures += 1; replacements += 1
        elif H >= T:                               # survives to T -> proactive swap
            cost += 10_000; hours += T; replacements += 1
        elif status == "Preventive":               # swapped early in reality (before T)
            cost += 10_000; hours += H; replacements += 1
        else:                                      # censored, still running, below T
            hours += H
    return {"T": T, "Failures": failures, "Replacements": replacements,
            "Total_Cost": cost, "Hours": hours, "Cost_per_1000h": cost / hours * 1000}

frontier = pandas.DataFrame([simulate(T) for T in range(3000, 10001, 250)])
run_to_failure = simulate(numpy.inf)
best = frontier.loc[frontier.Cost_per_1000h.idxmin()]
print(frontier.round(0).to_string(index=False))
print("\nRun-to-failure:", {k: round(v) for k, v in run_to_failure.items() if k != "T"})
print("\nBest policy:\n", best.round(0))

pyplot.figure()
pyplot.plot(frontier["T"], frontier["Cost_per_1000h"], marker="o")
pyplot.axhline(run_to_failure["Cost_per_1000h"], linestyle="--", color="red", label="Run to failure")
pyplot.axvline(best["T"], color="green", label=f"Minimum at T={best['T']:.0f} h")
pyplot.xlabel("Replacement threshold T (hours)"); pyplot.ylabel("Cost per 1,000 operating hours (USD)")
pyplot.title("Cost frontier"); pyplot.legend()
pyplot.savefig("cost_frontier.png", dpi=150, bbox_inches="tight")
pyplot.show()





#%%
# Step 6a: modeling setup (shared by all model cells)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, r2_score, mean_absolute_error

# Classification data: one row per machine-week. Target: did the needle fail this week?
# Leakage guard: only information known at the START of the week. Cost columns, replacement
# flags and end-of-week hours are excluded because they reveal the outcome.
features = ["Cumulative_Hours_Start", "Weekly_Working_Hours"] + sensors
X_week = pandas.get_dummies(DF[features + ["Season"]], columns=["Season"], drop_first=True, dtype=float)
y_week = DF["Failure"]

# Back test: train on the first 70% of weeks, test on the last 30% (time order, no shuffling,
# so the model never sees the future).
cut = DF["Week"].quantile(0.70)
train, test = DF["Week"] <= cut, DF["Week"] > cut
Xtr, Xte, ytr, yte = X_week[train], X_week[test], y_week[train], y_week[test]
print(f"Train weeks <= {cut:.0f}: {ytr.sum()} failures / {len(ytr)} rows; test: {yte.sum()} / {len(yte)}")

cv_clf = StratifiedKFold(n_splits=5, shuffle=True, random_state=297974)
clf_results = []

def evaluate_classifier(name, model):
    """5-fold CV AUC on train, then fit on train and score on the held-out future weeks."""
    cv_auc = cross_val_score(model, Xtr, ytr, cv=cv_clf, scoring="roc_auc")
    model.fit(Xtr, ytr)
    prob = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Xte)
    pred = model.predict(Xte)
    row = {"Model": name, "CV_AUC": float(cv_auc.mean()), "CV_AUC_sd": float(cv_auc.std()),
           "Test_AUC": roc_auc_score(yte, prob), "Test_Recall": recall_score(yte, pred),
           "Test_Precision": precision_score(yte, pred, zero_division=0), "Test_F1": f1_score(yte, pred)}
    clf_results.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    return model





#%%
# Model 1: logistic regression. class_weight="balanced" because failures are only ~1% of weeks
from sklearn.linear_model import LogisticRegression
logit = evaluate_classifier("Logistic regression",
    make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)))





#%%
# Model 2: PCA + logistic regression (PCA compresses the 10 correlated inputs to 4 components)
from sklearn.decomposition import PCA
pca_logit = evaluate_classifier("PCA(4) + logistic",
    make_pipeline(StandardScaler(), PCA(n_components=4), LogisticRegression(class_weight="balanced", max_iter=1000)))





#%%
# Model 3: LDA. No class_weight option; equal priors play the same role for the imbalance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = evaluate_classifier("LDA",
    make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(priors=[0.5, 0.5])))





#%%
# Model 4: LDA as dimension reduction + logistic regression (binary target -> 1 LDA component)
lda_logit = evaluate_classifier("LDA(1) + logistic",
    make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=1),
                  LogisticRegression(class_weight="balanced", max_iter=1000)))





#%%
# Model 5: SVC with RBF kernel. Scaling is essential for SVMs (see the practice set, exercise 2)
from sklearn.svm import SVC
svc = evaluate_classifier("SVC (RBF)",
    make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=297974)))





#%%
# Classification comparison table
clf_table = pandas.DataFrame(clf_results).round(3)
print(clf_table.to_string(index=False))





#%%
# Step 6b: regression setup. One row per FAILED needle (30). Target: hours at failure.
# Predictors: the needle's average operating conditions over its life.
failed_ids = needles.index[needles.Failed == 1]
life = (DF[DF.Needle_ID.isin(failed_ids)]
          .groupby("Needle_ID")[sensors + ["Weekly_Working_Hours"]].mean())
X_life = life
y_life = needles.loc[life.index, "Hours"]
cv_reg = KFold(n_splits=5, shuffle=True, random_state=297974)
reg_results = []

def evaluate_regressor(name, model):
    """30 needles are too few for a holdout split, so all evaluation is 5-fold cross-validation."""
    pred = cross_val_predict(model, X_life, y_life, cv=cv_reg)
    row = {"Model": name, "CV_R2": r2_score(y_life, pred), "CV_MAE_hours": mean_absolute_error(y_life, pred)}
    reg_results.append(row)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})





#%%
# Model 6: linear regression
from sklearn.linear_model import LinearRegression
evaluate_regressor("Linear regression", make_pipeline(StandardScaler(), LinearRegression()))





#%%
# Model 7: PCA + linear regression (principal component regression, like exam practice ex. 4)
evaluate_regressor("PCA(2) + linear (PCR)", make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression()))





#%%
# Model 8: PLS regression (supervised components, like Exam 1 question 16)
from sklearn.cross_decomposition import PLSRegression
evaluate_regressor("PLS (2 components)", make_pipeline(StandardScaler(), PLSRegression(n_components=2)))





#%%
# Model 9: SVR (RBF kernel, epsilon-insensitive tube)
from sklearn.svm import SVR
evaluate_regressor("SVR (RBF)", make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1000, epsilon=200)))





#%%
# Baseline: always predict the mean lifetime. A useful model must beat this.
from sklearn.dummy import DummyRegressor
evaluate_regressor("Baseline (mean)", DummyRegressor())
reg_table = pandas.DataFrame(reg_results).round(3)
print(reg_table.to_string(index=False))