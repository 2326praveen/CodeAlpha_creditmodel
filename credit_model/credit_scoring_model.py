"""
Train a credit scoring model using `data.csv` and the features requested by the user.

Features used:
- Age (demographic)
- Income (financial indicator)
- Debt (financial indicator)
- Credit_Score (financial/behavioral indicator)
- Payment_History (credit behavior, categorical: Good/Average/Bad)

The script trains a RandomForestClassifier, evaluates it on a hold-out test set and saves the trained model
to `best_credit_model.pkl` in the same folder.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

if os.path.exists('data.csv'):
    dataset_path = 'data.csv'
else:
    raise FileNotFoundError("data.csv not found in the current directory")


df = pd.read_csv(dataset_path)


required_cols = ['Age', 'Income', 'Debt', 'Credit_Score', 'Payment_History', 'Creditworthiness']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns in data.csv: {missing}")

X = df[['Age', 'Income', 'Debt', 'Credit_Score', 'Payment_History']].copy()
y = df['Creditworthiness'].values

ph_map = {'Bad': 0, 'Average': 1, 'Good': 2}
X['Payment_History'] = X['Payment_History'].map(ph_map).fillna(1).astype(int)


X['Age'] = X['Age'].fillna(X['Age'].median())
X['Income'] = X['Income'].fillna(X['Income'].median())
X['Debt'] = X['Debt'].fillna(X['Debt'].median())
X['Credit_Score'] = X['Credit_Score'].fillna(X['Credit_Score'].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

metrics = {
    'accuracy': accuracy_score(y_test, pred),
    'precision': precision_score(y_test, pred, zero_division=0),
    'recall': recall_score(y_test, pred, zero_division=0),
    'f1': f1_score(y_test, pred, zero_division=0),
    'roc_auc': roc_auc_score(y_test, proba) if proba is not None else None,
    'confusion_matrix': confusion_matrix(y_test, pred)
}

print("Model evaluation on test set:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
if metrics['roc_auc'] is not None:
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print("Confusion matrix:\n", metrics['confusion_matrix'])

model_path = 'best_credit_model.pkl'
joblib.dump(model, model_path)
print(f"Trained model saved to {model_path}")

print("\nTo use the model, run the Streamlit app: `streamlit run credit_app.py`")

