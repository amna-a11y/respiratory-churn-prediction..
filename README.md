# Customer Churn Prediction and Retention Analytics System

A complete machine learning project that cleans, analyzes, and models a Telco-style
customer dataset to predict churn and surface retention insights, with a
Streamlit dashboard on top.

## Project Structure

```
churn_project/
├── data/
│   ├── telco_customer_churn_raw.csv           # raw (messy, synthetic) input data
│   ├── telco_customer_churn_cleaned.csv        # cleaned + feature-engineered
│   └── telco_customer_churn_model_ready.csv    # encoded + scaled (model input)
├── notebooks/
│   └── Customer_Churn_Analysis.ipynb           # full walkthrough notebook
├── src/
│   ├── generate_dataset.py                     # builds the synthetic raw dataset
│   ├── data_pipeline.py                        # cleaning / FE / encoding / scaling functions
│   ├── eda.py                                  # generates all EDA charts
│   ├── train_models.py                         # trains & evaluates all 3 models
│   └── build_notebook.py                       # (dev tool) rebuilds the .ipynb file
├── models/                                      # saved trained models + encoders/scaler
├── outputs/                                     # saved charts, metrics, comparison table
├── app.py                                       # Streamlit dashboard
├── requirements.txt
└── README.md
```

## About the dataset

This was built to match the schema of the well-known **IBM Telco Customer Churn**
dataset from Kaggle (customerID, gender, tenure, Contract, MonthlyCharges,
Churn, etc.). Since this was generated offline without an internet connection,
`data/telco_customer_churn_raw.csv` is a **synthetic dataset** created by
`src/generate_dataset.py` that reproduces the same columns, same kinds of data-quality
issues (missing values, duplicates, inconsistent formatting, outliers), and the
same realistic churn drivers (month-to-month contracts, fiber-optic internet,
and electronic-check payments all churn more — same as the real data).

**If you have the real Kaggle CSV**, just download it and replace
`data/telco_customer_churn_raw.csv` with it (same column names) — every script
and notebook cell works unchanged.

## Setup (VS Code / local machine)

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the folder in VS Code**:
   ```bash
   code .
   ```
   Install the "Python" and "Jupyter" extensions if prompted.

## How to run everything

Run these from the project root, in order:

```bash
# 1. (Optional) regenerate the raw dataset — already included, skip if unchanged
python src/generate_dataset.py

# 2. Generate all EDA charts -> outputs/eda/
python src/eda.py

# 3. Clean data, engineer features, train & evaluate all 3 models -> models/, outputs/
python src/train_models.py

# 4. Launch the interactive dashboard
streamlit run app.py
```

Or open `notebooks/Customer_Churn_Analysis.ipynb` in VS Code / Jupyter and run
all cells top to bottom for the full narrated walkthrough (cleaning → EDA →
feature engineering → encoding → scaling → modeling → evaluation → business
insights).

## What's inside the notebook / scripts

- **Data Cleaning**: missing-value imputation (median for numeric, mode for
  categorical), duplicate removal, format normalization, outlier capping.
- **EDA**: churn distribution, histograms, boxplots, countplots (demographics,
  services, contracts, payment methods), correlation heatmap.
- **Feature Engineering**: tenure segments (New/Regular/Loyal), monthly-spend
  tiers (Low/Medium/High), total revenue (+ log transform), service count,
  contract risk score, average monthly spend.
- **Encoding**: One-Hot for nominal columns, Label Encoding for binary columns.
- **Scaling**: `StandardScaler` on the continuous numeric features.
- **Modeling**: Logistic Regression, Random Forest, and XGBoost (falls back
  automatically to `GradientBoostingClassifier` if `xgboost` isn't installed —
  no code changes needed either way).
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, 5-fold
  cross-validation, confusion matrices, ROC curves, precision-recall curve,
  feature importance.
- **Business insights**: which customer segments are highest-risk and
  concrete retention recommendations (see the last section of the notebook
  or `outputs/eda/eda_summary.txt`).

## Dashboard (`app.py`)

Three pages:
1. **Predict Churn** — fill in a customer profile, get a live churn
   probability from any of the three trained models plus suggested retention
   actions.
2. **Customer Insights** — churn-rate breakdowns by contract, payment method,
   internet service, and tenure, with a filterable data table.
3. **Model Performance** — side-by-side metrics table, ROC curves, confusion
   matrices, and feature importance charts for all three models.

## Notes

- `xgboost` and `streamlit` need to be installed locally (they're in
  `requirements.txt`) — they weren't available in the sandbox this project
  was assembled in, so `train_models.py` was verified here using the
  automatic `GradientBoostingClassifier` fallback. Once you `pip install
  xgboost`, running `train_models.py` again will train real XGBoost with no
  other changes required.
- All random seeds are fixed (`random_state=42`) for reproducibility.
