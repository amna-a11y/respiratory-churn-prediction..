"""
data_pipeline.py
-----------------
Reusable functions for the Customer Churn Prediction project:
    1. load_data
    2. clean_data
    3. engineer_features
    4. encode_features
    5. scale_features
    6. full_pipeline (runs all of the above in order)

Import these from the notebook or from train_models.py / app.py so the exact
same transformations are used everywhere (no train/serve skew).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# 2. CLEAN
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- fix inconsistent text formatting (whitespace / case) ---
    for col in ["gender", "Churn", "Partner", "Dependents", "PhoneService",
                "PaperlessBilling", "InternetService", "Contract", "PaymentMethod"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)

    df["gender"] = df["gender"].str.capitalize().replace({"Male": "Male", "Female": "Female"})
    df["Churn"] = df["Churn"].str.capitalize()

    # --- TotalCharges arrives as text with stray spaces / blanks in the real dataset ---
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # --- drop exact duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)

    # --- missing value imputation ---
    # Numerical -> median (robust to the outliers we know exist in MonthlyCharges)
    for col in ["MonthlyCharges", "TotalCharges"]:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    # Categorical -> mode
    for col in ["gender", "PaymentMethod"]:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)

    # --- outlier handling on MonthlyCharges (cap at 1.5*IQR, a handful of injected spikes) ---
    q1, q3 = df["MonthlyCharges"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    df["MonthlyCharges"] = np.where(df["MonthlyCharges"] > upper_bound, upper_bound, df["MonthlyCharges"])

    # --- basic sanity: tenure must be >= 0, Churn must be Yes/No ---
    df = df[df["tenure"] >= 0]
    df = df[df["Churn"].isin(["Yes", "No"])]

    df = df.reset_index(drop=True)
    df.attrs["rows_dropped_as_duplicates"] = dropped
    return df


# ---------------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

CONTRACT_RISK_MAP = {
    "Month-to-month": 3,
    "One year": 2,
    "Two year": 1,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Tenure segmentation ---
    def tenure_bucket(t):
        if t <= 12:
            return "New"
        elif t <= 48:
            return "Regular"
        else:
            return "Loyal"
    df["TenureSegment"] = df["tenure"].apply(tenure_bucket)

    # --- Monthly charges spend tier ---
    low_cut, high_cut = df["MonthlyCharges"].quantile([0.33, 0.66])
    def spend_bucket(v):
        if v <= low_cut:
            return "Low"
        elif v <= high_cut:
            return "Medium"
        else:
            return "High"
    df["SpendTier"] = df["MonthlyCharges"].apply(spend_bucket)

    # --- Total revenue feature (log-transform to tame skew, keep raw too) ---
    df["TotalRevenue"] = df["TotalCharges"]
    df["TotalRevenueLog"] = np.log1p(df["TotalRevenue"])

    # --- Service count: how many "Yes" services the customer subscribes to ---
    def count_services(row):
        return sum(1 for c in SERVICE_COLUMNS if row.get(c) == "Yes")
    df["ServiceCount"] = df.apply(count_services, axis=1)

    # --- Contract risk score (higher = more likely to churn) ---
    df["ContractRiskScore"] = df["Contract"].map(CONTRACT_RISK_MAP).fillna(3).astype(int)

    # --- Average revenue per month of tenure (avoids div-by-zero for tenure=0) ---
    df["AvgMonthlySpend"] = df["TotalRevenue"] / df["tenure"].replace(0, 1)

    return df


# ---------------------------------------------------------------------------
# 4. ENCODING
# ---------------------------------------------------------------------------
BINARY_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
NOMINAL_COLS = ["InternetService", "PaymentMethod", "Contract", "gender",
                 "TenureSegment", "SpendTier"]


def encode_features(df: pd.DataFrame, fit_encoders: bool = True, encoders: dict = None):
    """
    One-Hot encodes nominal columns, Label encodes binary columns.
    Returns (encoded_df, encoders) so the same encoders/columns can be reused at inference time.
    """
    df = df.copy()
    encoders = encoders or {}

    # Label encode binary Yes/No-style columns
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        if fit_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[f"le_{col}"] = le
        else:
            le = encoders[f"le_{col}"]
            df[col] = le.transform(df[col].astype(str))

    # Drop columns that are now redundant after feature engineering / not useful as raw features
    drop_cols = ["customerID"] + [c for c in SERVICE_COLUMNS if c not in ("PhoneService",)]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # One-hot encode nominal columns
    nominal_present = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=nominal_present, drop_first=True)

    return df, encoders


# ---------------------------------------------------------------------------
# 5. SCALING
# ---------------------------------------------------------------------------
SCALE_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "TotalRevenue",
              "TotalRevenueLog", "AvgMonthlySpend"]


def scale_features(df: pd.DataFrame, fit_scaler: bool = True, scaler: StandardScaler = None):
    df = df.copy()
    cols_present = [c for c in SCALE_COLS if c in df.columns]
    if fit_scaler:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
    else:
        df[cols_present] = scaler.transform(df[cols_present])
    return df, scaler


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
def full_pipeline(raw_csv_path: str):
    df_raw = load_data(raw_csv_path)
    df_clean = clean_data(df_raw)
    df_fe = engineer_features(df_clean)
    df_encoded, encoders = encode_features(df_fe, fit_encoders=True)
    df_scaled, scaler = scale_features(df_encoded, fit_scaler=True)
    return {
        "raw": df_raw,
        "clean": df_clean,
        "engineered": df_fe,
        "encoded": df_encoded,
        "scaled": df_scaled,
        "encoders": encoders,
        "scaler": scaler,
    }
