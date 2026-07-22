"""
train_models.py
-----------------
Trains and evaluates three classifiers on the processed churn dataset:
    1. Logistic Regression (interpretable baseline)
    2. Random Forest (ensemble, feature importances)
    3. XGBoost (gradient boosting; falls back to sklearn's
       GradientBoostingClassifier automatically if the xgboost package
       isn't installed, so this script runs anywhere)

Outputs:
    models/logistic_regression.pkl
    models/random_forest.pkl
    models/xgboost_model.pkl
    models/scaler.pkl
    models/encoders.pkl
    models/feature_columns.json
    outputs/model_comparison.csv
    outputs/*.png  (confusion matrices, ROC curves, feature importance)
"""

import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve,
)

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import full_pipeline

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "telco_customer_churn_raw.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def main():
    print("Running full data pipeline (clean -> engineer -> encode -> scale)...")
    pipe_out = full_pipeline(RAW_PATH)
    df = pipe_out["scaled"]

    # Save the fully processed dataset (a project deliverable)
    pipe_out["engineered"].to_csv(os.path.join(DATA_DIR, "telco_customer_churn_cleaned.csv"), index=False)
    df.to_csv(os.path.join(DATA_DIR, "telco_customer_churn_model_ready.csv"), index=False)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Churn rate - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }
    if HAS_XGBOOST:
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, n_jobs=-1
        )
    else:
        print("xgboost not installed in this environment -> using GradientBoostingClassifier "
              "as a drop-in substitute. `pip install xgboost` locally to use real XGBoost; "
              "no other code changes are needed.")
        models["XGBoost"] = GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        )

    results = []
    trained_models = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="roc_auc")

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4),
            "CV ROC-AUC (mean)": round(cv_scores.mean(), 4),
            "CV ROC-AUC (std)": round(cv_scores.std(), 4),
        })

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        fname = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(OUTPUTS_DIR, f"confusion_matrix_{fname}.png"), dpi=110)
        plt.close(fig)

        print(f"  Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  "
              f"F1={f1:.3f}  ROC-AUC={auc:.3f}  CV-AUC={cv_scores.mean():.3f}+/-{cv_scores.std():.3f}")

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    results_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"), index=False)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    # --- Combined ROC curve plot ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "roc_curves_comparison.png"), dpi=110)
    plt.close(fig)

    # --- Precision-Recall curve for the best model ---
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba_best)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_arr, prec_arr)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {best_name} (best model)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "precision_recall_best_model.png"), dpi=110)
    plt.close(fig)

    # --- Feature importance (Random Forest + best tree model) ---
    for name in ["Random Forest", "XGBoost"]:
        model = trained_models[name]
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(7, 6))
            importances.sort_values().plot(kind="barh", ax=ax, color="#3b6fb6")
            ax.set_title(f"Top 15 Feature Importances - {name}")
            ax.set_xlabel("Importance")
            plt.tight_layout()
            fname = name.lower().replace(" ", "_")
            plt.savefig(os.path.join(OUTPUTS_DIR, f"feature_importance_{fname}.png"), dpi=110)
            plt.close(fig)
            importances.sort_values(ascending=False).to_csv(os.path.join(OUTPUTS_DIR, f"feature_importance_{fname}.csv"))

    # --- Logistic Regression coefficients (interpretability) ---
    lr = trained_models["Logistic Regression"]
    coefs = pd.Series(lr.coef_[0], index=feature_columns).sort_values(key=abs, ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 6))
    coefs.sort_values().plot(kind="barh", ax=ax, color="#c0533b")
    ax.set_title("Top 15 Logistic Regression Coefficients (churn drivers)")
    ax.set_xlabel("Coefficient (impact on churn log-odds)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "logistic_regression_coefficients.png"), dpi=110)
    plt.close(fig)

    # --- Persist models + preprocessing artifacts for the Streamlit app ---
    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(trained_models["Logistic Regression"], f)
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(trained_models["Random Forest"], f)
    with open(os.path.join(MODELS_DIR, "xgboost_model.pkl"), "wb") as f:
        pickle.dump(trained_models["XGBoost"], f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(pipe_out["scaler"], f)
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(pipe_out["encoders"], f)
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(os.path.join(MODELS_DIR, "best_model_name.json"), "w") as f:
        json.dump({"best_model": best_name, "uses_real_xgboost": HAS_XGBOOST}, f, indent=2)

    print(f"\nBest model by ROC-AUC: {best_name}")
    print("All models and artifacts saved to models/. Plots saved to outputs/.")


if __name__ == "__main__":
    main()
