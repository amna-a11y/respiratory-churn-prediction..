"""
generate_dataset.py
--------------------
Generates a synthetic customer-churn dataset that follows the same schema
and statistical patterns as the well-known IBM Telco Customer Churn dataset.

Why synthetic? This environment has no internet access, so the original
Kaggle file can't be downloaded. This generator recreates the same columns,
value ranges, and relationships (e.g. month-to-month contracts churn far more
than two-year contracts) so every downstream step (cleaning, EDA, feature
engineering, modeling) behaves the same way it would on the real file.

If you have the real Kaggle CSV, just drop it in data/ as
`telco_customer_churn_raw.csv` with the same column names and skip this
script entirely.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 7043  # same size as the real IBM Telco dataset

def rand_choice(values, size, p=None):
    return np.random.choice(values, size=size, p=p)

# ---- base demographic / account fields -------------------------------------------------
customer_id = [f"{np.random.randint(1000,9999)}-{''.join(np.random.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}" for _ in range(N)]
gender = rand_choice(["Male", "Female"], N)
senior_citizen = rand_choice([0, 1], N, p=[0.84, 0.16])
partner = rand_choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = rand_choice(["Yes", "No"], N, p=[0.30, 0.70])

tenure = np.random.gamma(shape=2.0, scale=18, size=N).astype(int)
tenure = np.clip(tenure, 0, 72)

phone_service = rand_choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No", "No phone service",
    rand_choice(["Yes", "No"], N, p=[0.42, 0.58])
)

internet_service = rand_choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22])

def dependent_internet_feature(p_yes=0.4):
    out = []
    for svc in internet_service:
        if svc == "No":
            out.append("No internet service")
        else:
            out.append(np.random.choice(["Yes", "No"], p=[p_yes, 1 - p_yes]))
    return np.array(out)

online_security = dependent_internet_feature(0.35)
online_backup = dependent_internet_feature(0.40)
device_protection = dependent_internet_feature(0.40)
tech_support = dependent_internet_feature(0.35)
streaming_tv = dependent_internet_feature(0.45)
streaming_movies = dependent_internet_feature(0.45)

contract = rand_choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21])
paperless_billing = rand_choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = rand_choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

# ---- charges -----------------------------------------------------------------------------
base = np.where(internet_service == "Fiber optic", 70, np.where(internet_service == "DSL", 45, 20))
addon_cols = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
addon_cost = sum((col == "Yes").astype(int) for col in addon_cols) * np.random.uniform(4, 7, N)
phone_cost = np.where(phone_service == "Yes", np.random.uniform(15, 25, N), 0)

monthly_charges = np.round(base + addon_cost + phone_cost + np.random.normal(0, 3, N), 2)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 15, N), 2)
total_charges = np.clip(total_charges, 0, None)

# ---- churn (target) driven by realistic risk factors --------------------------------------
risk = (
    (contract == "Month-to-month") * 0.55
    + (contract == "One year") * 0.15
    + (internet_service == "Fiber optic") * 0.25
    + (payment_method == "Electronic check") * 0.20
    + (tenure < 12) * 0.30
    + (senior_citizen == 1) * 0.10
    + (paperless_billing == "Yes") * 0.10
    - (dependents == "Yes") * 0.10
    - (partner == "Yes") * 0.05
    - (tenure > 48) * 0.25
)
prob_churn = 1 / (1 + np.exp(-(risk - 1.0) * 3))
churn = np.where(np.random.rand(N) < prob_churn, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

# ---------------------------------------------------------------------------
# Inject realistic messiness so the "Data Cleaning" section has genuine work:
# ---------------------------------------------------------------------------
rng = np.random.default_rng(7)

# 1) Missing values in TotalCharges (mirrors the real dataset's known quirk
#    where brand-new customers have blank TotalCharges) + a few random NaNs elsewhere
new_cust_mask = df["tenure"] == 0
df.loc[new_cust_mask, "TotalCharges"] = np.nan
extra_missing_idx = rng.choice(df.index, size=40, replace=False)
df.loc[extra_missing_idx, "TotalCharges"] = np.nan

missing_monthly_idx = rng.choice(df.index, size=15, replace=False)
df.loc[missing_monthly_idx, "MonthlyCharges"] = np.nan

missing_gender_idx = rng.choice(df.index, size=20, replace=False)
df.loc[missing_gender_idx, "gender"] = np.nan

missing_payment_idx = rng.choice(df.index, size=25, replace=False)
df.loc[missing_payment_idx, "PaymentMethod"] = np.nan

# 2) Inconsistent categorical formatting (mixed case / stray whitespace)
inconsistent_idx = rng.choice(df.index, size=60, replace=False)
df.loc[inconsistent_idx, "gender"] = df.loc[inconsistent_idx, "gender"].apply(
    lambda v: v.lower() if isinstance(v, str) and rng.random() > 0.5 else (f" {v} " if isinstance(v, str) else v)
)
inconsistent_idx2 = rng.choice(df.index, size=40, replace=False)
df.loc[inconsistent_idx2, "Churn"] = df.loc[inconsistent_idx2, "Churn"].str.lower()

# 3) TotalCharges stored as strings with stray spaces (common real-world artifact)
str_mask = rng.choice(df.index, size=30, replace=False)
df["TotalCharges"] = df["TotalCharges"].astype(object)
df.loc[str_mask, "TotalCharges"] = df.loc[str_mask, "TotalCharges"].apply(
    lambda v: f" {v} " if pd.notna(v) else v
)

# 4) Duplicate rows
dup_rows = df.sample(25, random_state=1)
df = pd.concat([df, dup_rows], ignore_index=True)

# 5) A few outlier / anomalous values
outlier_idx = rng.choice(df.index, size=10, replace=False)
df.loc[outlier_idx, "MonthlyCharges"] = df.loc[outlier_idx, "MonthlyCharges"] * 4  # implausible spikes

# Shuffle rows so injected issues aren't clustered at the end
df = df.sample(frac=1, random_state=99).reset_index(drop=True)

df.to_csv("/home/claude/churn_project/data/telco_customer_churn_raw.csv", index=False)
print("Generated raw dataset:", df.shape)
print(df.isna().sum())
