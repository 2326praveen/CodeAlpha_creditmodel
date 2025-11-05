import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin


# PATH TO YOUR DATA (use file-local paths so script works when run from this folder)
BASE_DIR = Path(__file__).resolve().parent
# Prefer a user-provided 'data.csv' for custom training. Fall back to demo file if missing.
if (BASE_DIR / "data.csv").exists():
    DATA_PATH = BASE_DIR / "data.csv"
    USING_USER_DATA = True
else:
    DATA_PATH = BASE_DIR / "german_credit_data.csv"
    USING_USER_DATA = False
TARGET = "Risk"

# Load data
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found: {DATA_PATH}\nPlace your 'data.csv' or 'german_credit_data.csv' next to this script.")

df = pd.read_csv(DATA_PATH)

# Convert/validate target to 0/1. If the expected target column is missing
# and the user provided their own `data.csv`, try to auto-detect a binary
# target (e.g. 'Creditworthiness') before failing. For the demo file we
# still synthesize a simple label when missing.
if TARGET not in df.columns:
    if USING_USER_DATA:
        # Try to find common target column names automatically.
        candidate_names = ['Risk', 'risk', 'target', 'label', 'default', 'Default', 'y', 'creditworthiness']
        found = [c for c in candidate_names if c in df.columns]
        if found:
            TARGET = found[0]
            print(f"Found target column '{TARGET}' from common names.")
        else:
            # As a last resort, scan for any column that looks binary (2 unique values)
            binary_cols = []
            for c in df.columns:
                try:
                    nunique = df[c].dropna().nunique()
                except Exception:
                    nunique = None
                if nunique == 2:
                    # check values are 0/1 or common binary strings
                    vals = set(map(lambda v: str(v).lower(), df[c].dropna().unique()))
                    if vals.issubset({'0', '1'}) or vals.issubset({'yes', 'no'}) or vals.issubset({'good', 'bad'}) or vals.issubset({'true','false'}):
                        binary_cols.append(c)
            if len(binary_cols) == 1:
                TARGET = binary_cols[0]
                print(f"Auto-detected binary target column '{TARGET}'.")
            elif len(binary_cols) > 1:
                # prefer an obvious name if present
                prefer = [c for c in binary_cols if 'credit' in c.lower() or 'worth' in c.lower()]
                if prefer:
                    TARGET = prefer[0]
                    print(f"Multiple binary-looking columns found; picking '{TARGET}' based on name preference.")
                else:
                    TARGET = binary_cols[0]
                    print(f"Multiple binary-looking columns found; picking '{TARGET}' (first). If this is wrong, please specify target column explicitly.")
            else:
                raise ValueError("No target column found in data.csv. Please include a binary target column (e.g. 'Creditworthiness' or 'Risk' with values 0/1 or 'good'/'bad').")
    else:
        # No target provided in demo dataset — synthesize a simple numeric 0/1 label
        median_amt = df['Credit amount'].median()
        df[TARGET] = np.where(df['Credit amount'] > median_amt, 1, 0)
        print(f"Note: '{TARGET}' column not found. Created synthetic numeric target by Credit amount > {median_amt:.0f}.")

# At this point there should be a target column. Accept numeric 0/1 or two-class strings.
vals = df[TARGET].dropna().unique()
if pd.api.types.is_numeric_dtype(df[TARGET].dtype):
    unique_vals = set(map(int, np.unique(df[TARGET].dropna())))
    if not unique_vals.issubset({0, 1}):
        raise ValueError(f"Numeric target column '{TARGET}' must contain only 0 and 1 for binary classification. Found: {sorted(unique_vals)}")
    print(f"Using numeric target column '{TARGET}' as-is.")
else:
    # string-like targets: prefer mapping good/bad, otherwise map the two unique values
    vals_list = list(map(str, vals))
    vals_lower = [v.lower() for v in vals_list]
    if set(['good', 'bad']).issubset(set(vals_lower)):
        df[TARGET] = df[TARGET].map(lambda v: 0 if str(v).lower()=='good' else 1)
    else:
        # If exactly two classes, map first->0 second->1 but warn the user
        uniq = vals_list
        if len(uniq) == 2:
            print(f"Mapping target classes {uniq[0]} -> 0 and {uniq[1]} -> 1. If this is incorrect, relabel your data.")
            mapping = {uniq[0]: 0, uniq[1]: 1}
            df[TARGET] = df[TARGET].map(lambda v: mapping.get(str(v), np.nan))
        else:
            raise ValueError(f"Target column '{TARGET}' must be binary (2 classes). Found: {vals}")

# Ensure final target is numeric 0/1
if not pd.api.types.is_numeric_dtype(df[TARGET].dtype):
    df[TARGET] = df[TARGET].astype(int)

# Feature set: if the user provided `data.csv`, use the requested features.
def _find_actual_columns(df, desired_list):
    """Map a list of desired feature names to actual dataframe column names.
    This helps handle differences like spaces vs underscores.
    """
    actual = []
    cols = list(df.columns)
    def _norm(s):
        return ''.join(ch for ch in str(s).lower() if ch.isalnum())

    norm_map = {c: _norm(c) for c in cols}
    for d in desired_list:
        dn = _norm(d)
        matched = [c for c, nc in norm_map.items() if nc == dn]
        if matched:
            actual.append(matched[0])
        else:
            # try partial match (e.g., creditscore -> creditscore)
            partial = [c for c, nc in norm_map.items() if dn in nc or nc in dn]
            if partial:
                actual.append(partial[0])
            else:
                raise KeyError(f"Requested feature '{d}' not found in data columns. Available columns: {cols}")
    return actual

if USING_USER_DATA:
    # desired features (user requested): Age, Income, Debt, Credit Score, Payment History
    desired_numeric = ["Age", "Income", "Debt", "Credit Score"]
    desired_categorical = ["Payment History"]
    NUMERIC_FEATURES = _find_actual_columns(df, desired_numeric)
    CATEGORICAL_FEATURES = _find_actual_columns(df, desired_categorical)
else:
    # Keep original demo features for the provided german_credit_data.csv
    NUMERIC_FEATURES = ["Age", "Credit amount", "Duration"]
    CATEGORICAL_FEATURES = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

# ============ Feature engineering (kept inside pipeline) ============
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
        # nothing to fit
        return self

    def transform(self, X):
        Xc = X.copy()
        # Avoid division by zero
        denom = (Xc[self.income_col].astype(float) + 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            Xc[self.new_col] = Xc[self.debt_col].astype(float) / denom
        Xc[self.new_col] = Xc[self.new_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return Xc

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])

# Encode Payment_History with an ordinal meaning: Bad < Average < Good
payment_order = [["Bad", "Average", "Good"]]
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=payment_order, handle_unknown="use_encoded_value", unknown_value=-1))
])

# We'll compute DTI in a pipeline step before the ColumnTransformer
DTI_FEATURE = "DTI"
FULL_NUMERIC_FEATURES = NUMERIC_FEATURES + [DTI_FEATURE]

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, FULL_NUMERIC_FEATURES),
    ("cat", categorical_pipe, CATEGORICAL_FEATURES)
])

# Base models with imbalance handling where supported
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced_subsample"),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
}

best_auc = -np.inf
best_model = None
best_name = ""

for name, model in models.items():
    pipe = Pipeline([
        ("feat", AddDTI(income_col=NUMERIC_FEATURES[1], debt_col=NUMERIC_FEATURES[2], new_col=DTI_FEATURE)),
        ("pre", preprocessor),
        ("clf", model)
    ])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)

    print(f"{name} ROC-AUC: {auc:.3f}")
    print(classification_report(y_test, pipe.predict(X_test)))

    if auc > best_auc:
        best_model = pipe
        best_auc = auc
        best_name = name

# ================== Hyperparameter tuning (lightweight) ==================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define search spaces
param_spaces = {
    "LogisticRegression": {
        "clf__C": np.logspace(-3, 2, 20)
    },
    "RandomForest": {
        "clf__n_estimators": [200, 300, 400, 500, 600],
        "clf__max_depth": [None, 5, 10, 15, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8, 12],
        "clf__max_features": ["sqrt", "log2", None]
    },
    "HistGradientBoosting": {
        "clf__learning_rate": np.linspace(0.02, 0.2, 10),
        "clf__max_depth": [3, 5, 10, None],
        "clf__min_samples_leaf": [1, 5, 10, 20]
    }
}

for name, base_model in models.items():
    print(f"\nTuning {name}...")
    base_pipe = Pipeline([
        ("feat", AddDTI(income_col=NUMERIC_FEATURES[1], debt_col=NUMERIC_FEATURES[2], new_col=DTI_FEATURE)),
        ("pre", preprocessor),
        ("clf", base_model)
    ])
    params = param_spaces.get(name, {})
    if not params:
        # skip tuning if no params defined
        tuned_pipe = base_pipe
        tuned_pipe.fit(X_train, y_train)
        proba = tuned_pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        score_info = "(no tuning)"
    else:
        search = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=params,
            n_iter=15,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)
        tuned_pipe = search.best_estimator_
        proba = tuned_pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        score_info = f"(cv best {search.best_score_:.3f})"

    print(f"{name} tuned ROC-AUC: {auc:.3f} {score_info}")
    print(classification_report(y_test, tuned_pipe.predict(X_test)))

    if auc > best_auc:
        best_model = tuned_pipe
        best_auc = auc
        best_name = f"{name} (tuned)"

# Save artifacts (inside repo folder `artifacts`)
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)
joblib.dump(best_model, ARTIFACTS_DIR / "best_model.pkl")

schema = {
    "numeric": NUMERIC_FEATURES,  # keep only base inputs for the UI
    "categorical": CATEGORICAL_FEATURES,
    "target": TARGET,
    "best_model": best_name,
    "roc_auc": float(best_auc)
}

# Add observed categorical levels to schema to help the UI build selectboxes
categories = {}
for c in CATEGORICAL_FEATURES:
    if c in df.columns:
        # preserve order and remove NaNs
        cats = list(pd.Series(df[c].dropna().unique()))
    else:
        cats = []
    categories[c] = cats

schema["categories"] = categories

with open(ARTIFACTS_DIR / "schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print(f"\n✅ Best Model: {best_name} (AUC={best_auc:.3f}) saved!")
