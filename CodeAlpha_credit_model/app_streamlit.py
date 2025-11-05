import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer needed to load the saved model
class AddDTI(BaseEstimator, TransformerMixin):
    """
    Adds 'DTI' = Debt / (Income + 1) to the dataframe, leaving other columns intact.
    Works with pandas DataFrame inputs.
    """
    def __init__(self, income_col: str, debt_col: str, new_col: str = "DTI"):
        self.income_col = income_col
        self.debt_col = debt_col
        self.new_col = new_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X.copy()
        denom = (Xc[self.income_col].astype(float) + 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            Xc[self.new_col] = Xc[self.debt_col].astype(float) / denom
        Xc[self.new_col] = Xc[self.new_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xc

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
model_path = ARTIFACTS_DIR / "best_model.pkl"
schema_path = ARTIFACTS_DIR / "schema.json"

if not model_path.exists() or not schema_path.exists():
    st.error("Model artifacts not found. Run `train_credit_model.py` first to produce the model and schema in the `artifacts` folder.")
    st.stop()

model = joblib.load(model_path)
schema = json.load(open(schema_path))

num_cols = schema.get("numeric", [])
cat_cols = schema.get("categorical", [])
categories = schema.get("categories", {})

st.title("💳 Credit Scoring")
st.markdown(f"**Model:** {schema.get('best_model','(unknown)')} — AUC: {schema.get('roc_auc',0):.3f}")

st.sidebar.header("Input features")
inputs = {}

# Numeric inputs with reasonable defaults
for col in num_cols:
    if col == "Age":
        default = 30
    elif col.lower().startswith("income"):
        default = 50000
    elif col.lower().startswith("debt"):
        default = 5000
    elif col.lower().startswith("credit"):
        default = 600
    else:
        default = 0
    inputs[col] = st.sidebar.number_input(col, value=default)

# Categorical / Payment History
for col in cat_cols:
    opts = categories.get(col, [])
    if opts:
        # convert any non-string options to str for display
        opts_str = [str(o) for o in opts]
        sel = st.sidebar.selectbox(col, options=opts_str)
        inputs[col] = sel
    else:
        inputs[col] = st.sidebar.text_input(col, "")

if st.sidebar.button("Predict"):
    df = pd.DataFrame([inputs])
    try:
        proba = model.predict_proba(df)[0][1]
        pred = model.predict(df)[0]
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    result = "Bad Credit Risk" if int(pred) == 1 else "Good Credit Risk"
    st.subheader(f"Prediction: {result}")
    st.write(f"Probability of Default: **{proba:.2f}**")
