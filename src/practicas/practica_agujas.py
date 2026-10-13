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
