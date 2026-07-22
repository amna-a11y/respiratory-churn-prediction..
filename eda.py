"""
eda.py
-------
Generates the Exploratory Data Analysis visualizations required by the
project spec: histograms, boxplots, countplots, a correlation heatmap,
and a churn distribution chart, plus demographic / usage / contract /
payment-behavior breakdowns. All figures are saved to outputs/eda/.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import load_data, clean_data, engineer_features

sns.set_theme(style="whitegrid")

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_customer_churn_raw.csv")
EDA_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "eda")
os.makedirs(EDA_DIR, exist_ok=True)


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, name), dpi=110)
    plt.close()


def main():
    df = engineer_features(clean_data(load_data(RAW_PATH)))

    # 1) Churn distribution
    plt.figure(figsize=(5, 4))
    ax = sns.countplot(data=df, x="Churn", hue="Churn", palette=["#3b6fb6", "#c0533b"], legend=False)
    ax.set_title("Churn Distribution")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    savefig("01_churn_distribution.png")

    # 2) Histograms - numeric distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.histplot(df[col], bins=30, kde=True, ax=ax, color="#3b6fb6")
        ax.set_title(f"Distribution of {col}")
    savefig("02_histograms_numeric.png")

    # 3) Boxplots - numeric vs churn
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.boxplot(data=df, x="Churn", y=col, hue="Churn", ax=ax, palette=["#3b6fb6", "#c0533b"], legend=False)
        ax.set_title(f"{col} by Churn")
    savefig("03_boxplots_by_churn.png")

    # 4) Countplots - demographics vs churn
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    demo_cols = ["gender", "SeniorCitizen", "Partner", "Dependents"]
    for ax, col in zip(axes.flatten(), demo_cols):
        sns.countplot(data=df, x=col, hue="Churn", ax=ax, palette=["#3b6fb6", "#c0533b"])
        ax.set_title(f"Churn by {col}")
    savefig("04_demographics_vs_churn.png")

    # 5) Service usage analysis
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    service_cols = ["InternetService", "OnlineSecurity", "TechSupport",
                     "StreamingTV", "StreamingMovies", "ServiceCount"]
    for ax, col in zip(axes.flatten(), service_cols):
        if col == "ServiceCount":
            sns.boxplot(data=df, x="Churn", y=col, hue="Churn", ax=ax, palette=["#3b6fb6", "#c0533b"], legend=False)
        else:
            sns.countplot(data=df, x=col, hue="Churn", ax=ax, palette=["#3b6fb6", "#c0533b"])
        ax.set_title(f"Churn by {col}")
        ax.tick_params(axis="x", rotation=20)
    savefig("05_service_usage_vs_churn.png")

    # 6) Contract & subscription analysis
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col in zip(axes, ["Contract", "PaperlessBilling", "TenureSegment"]):
        sns.countplot(data=df, x=col, hue="Churn", ax=ax, palette=["#3b6fb6", "#c0533b"])
        ax.set_title(f"Churn by {col}")
        ax.tick_params(axis="x", rotation=15)
    savefig("06_contract_subscription_vs_churn.png")

    # 7) Payment behavior analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=axes[0], palette=["#3b6fb6", "#c0533b"])
    axes[0].set_title("Churn by Payment Method")
    axes[0].tick_params(axis="x", rotation=20)
    sns.countplot(data=df, x="SpendTier", hue="Churn", ax=axes[1], palette=["#3b6fb6", "#c0533b"])
    axes[1].set_title("Churn by Spend Tier")
    savefig("07_payment_behavior_vs_churn.png")

    # 8) Correlation heatmap (numeric features only)
    numeric_df = df.copy()
    numeric_df["Churn_flag"] = (numeric_df["Churn"] == "Yes").astype(int)
    numeric_df["SeniorCitizen"] = numeric_df["SeniorCitizen"].astype(int)
    corr_cols = ["tenure", "MonthlyCharges", "TotalCharges", "TotalRevenueLog",
                 "ServiceCount", "ContractRiskScore", "AvgMonthlySpend",
                 "SeniorCitizen", "Churn_flag"]
    corr = numeric_df[corr_cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap - Numeric & Engineered Features")
    savefig("08_correlation_heatmap.png")

    # --- Text summary of key insights (also used in README/report) ---
    churn_rate = (df["Churn"] == "Yes").mean()
    by_contract = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
    by_internet = df.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
    by_payment = df.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)

    summary = []
    summary.append(f"Overall churn rate: {churn_rate:.1%}\n")
    summary.append("Churn rate by contract type:\n" + by_contract.to_string() + "\n")
    summary.append("Churn rate by internet service:\n" + by_internet.to_string() + "\n")
    summary.append("Churn rate by payment method:\n" + by_payment.to_string() + "\n")

    with open(os.path.join(EDA_DIR, "eda_summary.txt"), "w") as f:
        f.write("\n".join(summary))

    print("EDA complete. Figures saved to outputs/eda/")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
