
Streamlit Credit Scoring Web App

Accepts Age, Income, Debt, Credit Score, Payment History and predicts creditworthiness
using the trained model saved as `best_credit_model.pkl`.
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os


st.sidebar.title("Credit Scoring Model")
st.sidebar.info("Predict creditworthiness. Provide the applicant's Age, Income, Debt, Credit Score and Payment History.")

# Load model
model_path = 'best_credit_model.pkl'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please run the training script first.")
    st.stop()

model = joblib.load(model_path)

# Feature names used during training
FEATURE_NAMES = ['Age', 'Income', 'Debt', 'Credit_Score', 'Payment_History']

st.title("Creditworthiness Prediction")
st.write("Enter applicant details below:")

# Input fields as requested
age = st.number_input("Age", min_value=16, max_value=100, value=35, step=1)
income = st.number_input("Income ($)", min_value=0.0, value=50000.0, step=100.0)
debt = st.number_input("Debt ($)", min_value=0.0, value=10000.0, step=100.0)
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=600, step=1)
payment_history = st.selectbox("Payment History", options=["Good", "Average", "Bad"], index=1)

st.markdown("""
**To run this app:**
```
streamlit run credit_app.py
```
Then open http://localhost:8501 in your browser.
""")

if st.button("Predict"):
    # Map payment history to numeric same as training
    ph_map = {'Bad': 0, 'Average': 1, 'Good': 2}
    ph_val = ph_map.get(payment_history, 1)

    input_df = pd.DataFrame([[age, income, debt, credit_score, ph_val]], columns=FEATURE_NAMES)

    # Predict
    try:
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
        pred = int(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        raise

    if pred == 1:
        st.success("✅ Result: CREDITWORTHY")
        st.write("This applicant is predicted to be creditworthy.")
    else:
        st.error("❌ Result: NOT CREDITWORTHY")
        st.write("This applicant is predicted NOT to be creditworthy.")

    if prob is not None:
        st.write(f"**Confidence:** {prob*100:.1f}%")
        st.progress(float(prob))

        if prob > 0.8:
            st.info("🟢 High confidence")
        elif prob > 0.6:
            st.info("🟡 Moderate confidence")
        elif prob > 0.4:
            st.warning("🟠 Low confidence")
        else:
            st.warning("🔴 Very low confidence")

    # Additional quick checks
    dti = (debt / income) * 100 if income > 0 else 0
    st.write(f"**Debt-to-Income Ratio:** {dti:.1f}%")
    if dti > 40:
        st.warning("⚠️ High debt-to-income ratio (>40%)")

    # Show top feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        st.write("**Feature importances:**")
        for i in order:
            st.write(f"- {FEATURE_NAMES[i]}: {importances[i]:.3f}")

