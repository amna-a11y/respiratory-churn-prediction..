"""
app.py
-------
Streamlit dashboard for the Customer Churn Prediction & Retention Analytics
project. Three pages:
    1. Predict Churn     - interactive form, scores a single customer live
    2. Customer Insights - EDA charts (churn drivers, demographics, contracts)
    3. Model Performance  - comparison table + confusion matrices / ROC curves

Run with:
    streamlit run app.py
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from data_pipeline import clean_data, engineer_features, encode_features, scale_features, SERVICE_COLUMNS

BASE = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "models")
DATA_DIR = os.path.join(BASE, "data")
OUTPUTS_DIR = os.path.join(BASE, "outputs")

st.set_page_config(page_title="Churn Analytics | Retention Intelligence", layout="wide", page_icon="📊")

# ---------------------------------------------------------------------------
# Design system: color palette + global CSS
# ---------------------------------------------------------------------------
NAVY = "#0F2440"
NAVY_LIGHT = "#1E3A5F"
ACCENT_BLUE = "#2563EB"
ACCENT_TEAL = "#0D9488"
DANGER = "#DC2626"
SUCCESS = "#16A34A"
WARN_BG = "#FFF7ED"
SLATE = "#475569"
BG_CARD = "#F8FAFC"
BORDER = "#E2E8F0"

CHURN_PALETTE = {"No": ACCENT_TEAL, "Yes": DANGER}
CHURN_PALETTE_LIST = [ACCENT_TEAL, DANGER]

sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": BORDER,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": SLATE,
    "ytick.color": SLATE,
    "axes.titleweight": "bold",
    "font.family": "sans-serif",
})
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    /* Top banner */
    .brand-banner {{
        background: linear-gradient(120deg, {NAVY} 0%, {NAVY_LIGHT} 55%, {ACCENT_TEAL} 130%);
        border-radius: 14px;
        padding: 28px 32px;
        margin-bottom: 28px;
        box-shadow: 0 4px 18px rgba(15, 36, 64, 0.18);
    }}
    .brand-banner h1 {{
        color: #FFFFFF;
        font-size: 1.65rem;
        font-weight: 800;
        margin: 0 0 4px 0;
        letter-spacing: -0.02em;
    }}
    .brand-banner p {{
        color: #CBD9EA;
        font-size: 0.95rem;
        margin: 0;
        font-weight: 400;
    }}

    /* Section headers */
    .section-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {NAVY};
        margin: 0 0 2px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-sub {{
        color: {SLATE};
        font-size: 0.88rem;
        margin-bottom: 14px;
    }}

    /* KPI cards */
    .kpi-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
    }}
    .kpi-label {{
        color: {SLATE};
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: {NAVY};
        font-size: 1.7rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}

    /* Risk result card */
    .risk-card {{
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 6px;
        border: 1px solid {BORDER};
    }}
    .risk-card.high {{
        background: #FEF2F2;
        border-color: #FCA5A5;
    }}
    .risk-card.low {{
        background: #F0FDF4;
        border-color: #86EFAC;
    }}
    .risk-title {{
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: {SLATE};
        margin-bottom: 4px;
    }}
    .risk-value {{
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}
    .risk-value.high {{ color: {DANGER}; }}
    .risk-value.low {{ color: {SUCCESS}; }}
    .risk-tag {{
        display: inline-block;
        margin-top: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
    }}
    .risk-tag.high {{ background: {DANGER}; color: white; }}
    .risk-tag.low {{ background: {SUCCESS}; color: white; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {NAVY};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E2E8F0 !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-weight: 500;
    }}
    section[data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.15);
    }}

    /* Buttons */
    .stButton > button {{
        background: {ACCENT_BLUE};
        color: white;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.55rem 1.4rem;
    }}
    .stButton > button:hover {{
        background: {NAVY_LIGHT};
        color: white;
    }}

    /* DataFrames */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}

    hr {{ margin: 1.6rem 0; }}
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle=None, icon=""):
    st.markdown(f'<div class="section-title">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)

    models = {}
    for name, fname in [("Logistic Regression", "logistic_regression.pkl"),
                         ("Random Forest", "random_forest.pkl"),
                         ("XGBoost", "xgboost_model.pkl")]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)

    best_name = "Random Forest"
    best_path = os.path.join(MODELS_DIR, "best_model_name.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            best_name = json.load(f).get("best_model", best_name)

    return scaler, encoders, feature_columns, models, best_name


@st.cache_data
def load_clean_dataset():
    path = os.path.join(DATA_DIR, "telco_customer_churn_cleaned.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    raw = pd.read_csv(os.path.join(DATA_DIR, "telco_customer_churn_raw.csv"))
    return engineer_features(clean_data(raw))


@st.cache_data
def load_model_comparison():
    path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def build_single_customer_row(inputs: dict) -> pd.DataFrame:
    """Turn form inputs into a one-row DataFrame matching the raw schema."""
    row = {
        "customerID": "0000-NEWUI",
        "gender": inputs["gender"],
        "SeniorCitizen": 1 if inputs["senior"] == "Yes" else 0,
        "Partner": inputs["partner"],
        "Dependents": inputs["dependents"],
        "tenure": inputs["tenure"],
        "PhoneService": inputs["phone_service"],
        "MultipleLines": inputs["multiple_lines"],
        "InternetService": inputs["internet_service"],
        "OnlineSecurity": inputs["online_security"],
        "OnlineBackup": inputs["online_backup"],
        "DeviceProtection": inputs["device_protection"],
        "TechSupport": inputs["tech_support"],
        "StreamingTV": inputs["streaming_tv"],
        "StreamingMovies": inputs["streaming_movies"],
        "Contract": inputs["contract"],
        "PaperlessBilling": inputs["paperless"],
        "PaymentMethod": inputs["payment_method"],
        "MonthlyCharges": inputs["monthly_charges"],
        "TotalCharges": inputs["monthly_charges"] * max(inputs["tenure"], 1),
        "Churn": "No",  # placeholder, dropped before prediction
    }
    return pd.DataFrame([row])


def predict_single_customer(raw_row: pd.DataFrame, scaler, encoders, feature_columns, model):
    df_clean = clean_data(raw_row)
    df_fe = engineer_features(df_clean)
    df_encoded, _ = encode_features(df_fe, fit_encoders=False, encoders=encoders)
    df_scaled, _ = scale_features(df_encoded, fit_scaler=False, scaler=scaler)

    X = df_scaled.drop(columns=["Churn"])
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    proba = model.predict_proba(X)[0, 1]
    pred = model.predict(X)[0]
    return pred, proba


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 📊 Churn Analytics")
st.sidebar.caption("Customer Churn Prediction & Retention Analytics System")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Predict Churn", "Customer Insights", "Model Performance"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Built with scikit-learn, XGBoost & Streamlit")

try:
    scaler, encoders, feature_columns, models, best_name = load_artifacts()
    artifacts_ready = len(models) > 0
except FileNotFoundError:
    artifacts_ready = False

if not artifacts_ready:
    st.sidebar.error("Model artifacts not found. Run `python src/train_models.py` first.")


# ---------------------------------------------------------------------------
# PAGE 1: Predict Churn
# ---------------------------------------------------------------------------
if page == "Predict Churn":
    st.markdown("""
    <div class="brand-banner">
        <h1>Predict Customer Churn</h1>
        <p>Score a customer profile in real time and get suggested retention actions.</p>
    </div>
    """, unsafe_allow_html=True)

    if not artifacts_ready:
        st.warning("Train the models first: `python src/train_models.py`")
    else:
        model_choice = st.selectbox("Prediction model", list(models.keys()),
                                     index=list(models.keys()).index(best_name) if best_name in models else 0)

        col1, col2, col3 = st.columns(3)
        with col1:
            section_header("Demographics", icon="👤")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            dependents = st.selectbox("Has Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            section_header("Services", icon="🌐")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        with col3:
            section_header("Account", icon="💳")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)

        st.write("")
        predict_clicked = st.button("Predict Churn Risk", type="primary")

        if predict_clicked:
            inputs = dict(gender=gender, senior=senior, partner=partner, dependents=dependents,
                          tenure=tenure, phone_service=phone_service, multiple_lines=multiple_lines,
                          internet_service=internet_service, online_security=online_security,
                          online_backup=online_backup, device_protection=device_protection,
                          tech_support=tech_support, streaming_tv=streaming_tv,
                          streaming_movies=streaming_movies, contract=contract, paperless=paperless,
                          payment_method=payment_method, monthly_charges=monthly_charges)
            raw_row = build_single_customer_row(inputs)
            model = models[model_choice]
            pred, proba = predict_single_customer(raw_row, scaler, encoders, feature_columns, model)

            risk_class = "high" if proba >= 0.5 else "low"
            risk_text = "High Risk of Churn" if proba >= 0.5 else "Likely to Stay"

            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <div class="risk-title">Churn Probability &middot; {model_choice}</div>
                <div class="risk-value {risk_class}">{proba:.1%}</div>
                <span class="risk-tag {risk_class}">{risk_text}</span>
            </div>
            """, unsafe_allow_html=True)
            st.progress(min(max(proba, 0.0), 1.0))

            if proba >= 0.5:
                st.info(
                    "**Suggested retention actions**: consider offering a contract upgrade "
                    "incentive, switching billing to autopay, or a loyalty discount — "
                    "especially if this customer is on a month-to-month contract or "
                    "electronic check payment."
                )


# ---------------------------------------------------------------------------
# PAGE 2: Customer Insights
# ---------------------------------------------------------------------------
elif page == "Customer Insights":
    st.markdown("""
    <div class="brand-banner">
        <h1>Customer Insights</h1>
        <p>Explore churn patterns across demographics, services, contracts, and payment behavior.</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_clean_dataset()
    churn_rate = (df["Churn"] == "Yes").mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Total Customers", f"{len(df):,}")
    with c2: kpi_card("Churn Rate", f"{churn_rate:.1%}")
    with c3: kpi_card("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
    with c4: kpi_card("Avg Tenure", f"{df['tenure'].mean():.1f} mo")

    st.write("")
    st.markdown("---")

    left, right = st.columns(2)
    with left:
        section_header("Churn by Contract Type")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="Contract", hue="Churn", ax=ax, palette=CHURN_PALETTE)
        ax.legend(title="Churn", frameon=False)
        st.pyplot(fig, use_container_width=True)

    with right:
        section_header("Churn by Payment Method")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=ax, palette=CHURN_PALETTE)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Churn", frameon=False)
        st.pyplot(fig, use_container_width=True)

    left2, right2 = st.columns(2)
    with left2:
        section_header("Tenure Distribution by Churn")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x="Churn", y="tenure", hue="Churn", ax=ax, palette=CHURN_PALETTE, legend=False)
        st.pyplot(fig, use_container_width=True)

    with right2:
        section_header("Churn by Internet Service")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax, palette=CHURN_PALETTE)
        ax.legend(title="Churn", frameon=False)
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    section_header("Filter & Explore", icon="🔎")
    contract_f = st.multiselect("Contract", df["Contract"].unique().tolist(), default=df["Contract"].unique().tolist())
    filtered = df[df["Contract"].isin(contract_f)]
    st.dataframe(filtered.head(200), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE 3: Model Performance
# ---------------------------------------------------------------------------
elif page == "Model Performance":
    st.markdown("""
    <div class="brand-banner">
        <h1>Model Performance</h1>
        <p>Compare Logistic Regression, Random Forest, and XGBoost on accuracy, recall, and ROC-AUC.</p>
    </div>
    """, unsafe_allow_html=True)

    comparison = load_model_comparison()
    if comparison is None:
        st.warning("Run `python src/train_models.py` to generate model comparison results.")
    else:
        section_header("Model Comparison", icon="📋")
        st.dataframe(comparison, use_container_width=True)

        best_row = comparison.sort_values("ROC-AUC", ascending=False).iloc[0]
        st.success(f"Best model by ROC-AUC: **{best_row['Model']}** (ROC-AUC = {best_row['ROC-AUC']:.3f})")

        st.markdown("---")
        section_header("ROC Curve Comparison", icon="📈")
        roc_path = os.path.join(OUTPUTS_DIR, "roc_curves_comparison.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)

        section_header("Confusion Matrices", icon="🧮")
        cols = st.columns(3)
        for col, name in zip(cols, ["logistic_regression", "random_forest", "xgboost"]):
            path = os.path.join(OUTPUTS_DIR, f"confusion_matrix_{name}.png")
            if os.path.exists(path):
                col.image(path, use_container_width=True)

        section_header("Feature Importance", icon="⭐")
        fi_col1, fi_col2 = st.columns(2)
        rf_path = os.path.join(OUTPUTS_DIR, "feature_importance_random_forest.png")
        xgb_path = os.path.join(OUTPUTS_DIR, "feature_importance_xgboost.png")
        if os.path.exists(rf_path):
            fi_col1.image(rf_path, use_container_width=True)
        if os.path.exists(xgb_path):
            fi_col2.image(xgb_path, use_container_width=True)