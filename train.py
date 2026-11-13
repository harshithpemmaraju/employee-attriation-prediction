"""
train.py - Trains the Employee Attrition model and saves it to model.pkl

Workflow copied from the notebook (employee_attriation_project.ipynb):
  * Target column ........ Attrition (1 = employee left, 0 = employee stayed)
  * Split ................ 80/20, random_state=42, stratified on the target
  * Numeric features ..... median imputation -> StandardScaler
  * Categorical features . most-frequent imputation -> OneHotEncoder(handle_unknown="ignore")
  * Model ................ Logistic Regression with the tuned hyper-parameters
                           found in the notebook (C=0.1, class_weight=None, max_iter=1000)

Why Logistic Regression? In the notebook it had the best cross-validated macro-F1
(0.8659) after tuning and the best F1-score on the held-out test set (0.8208).
Gradient Boosting was almost tied (0.8189), so the choice is close.

Run:   python train.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "employee_attrition.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
TARGET = "Attrition"

# Settings taken from the notebook (Step 4 - Hyperparameter Tuning, best Logistic Regression)
TEST_SIZE = 0.20
RANDOM_STATE = 42
BEST_PARAMS = {"C": 0.1, "class_weight": None, "max_iter": 1000, "random_state": RANDOM_STATE}


def main() -> None:
    # ------------------------------------------------------------------ load
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: dataset not found at {DATA_PATH}. Keep employee_attrition.csv next to train.py.")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    if TARGET not in df.columns:
        sys.exit(f"ERROR: target column '{TARGET}' not found in the dataset.")

    # The notebook found 0 duplicate rows; this keeps the script safe if the CSV ever changes.
    n_dupes = int(df.duplicated().sum())
    if n_dupes:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Removed {n_dupes} duplicate rows")

    # ------------------------------------------------------- X / y and columns
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # Numeric = number dtype, everything else = categorical (works on pandas 1.x, 2.x and 3.x)
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

    # --------------------------------------------------- stratified split first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ------------------------------------------------------------- the pipeline
    # Every step below is fitted on the TRAINING data only (no leakage).
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),  # unseen categories -> all zeros
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("model", LogisticRegression(**BEST_PARAMS))]
    )

    # ------------------------------------------------------------------- train
    pipeline.fit(X_train, y_train)

    # ---------------------------------------------------------------- evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    print("\n=== Test-set results (20% hold-out, never used for fitting) ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed (0)", "Left (1)"]))

    # ------------------------------------------- information the app needs
    # Allowed values / ranges come from the training data so the app never has to guess them.
    numeric_info = {}
    for col in numeric_cols:
        s = X_train[col].dropna()
        numeric_info[col] = {
            "min": float(s.min()),
            "max": float(s.max()),
            "median": float(s.median()),
            "is_integer": bool(np.allclose(s, s.round())),
        }
    categories = {col: sorted(X_train[col].dropna().unique().tolist()) for col in categorical_cols}

    # In this dataset every JobRole belongs to exactly one Department. The app uses this
    # mapping to stop impossible combinations (e.g. "Software Engineer" in "Finance").
    department_roles = {}
    if {"Department", "JobRole"} <= set(X_train.columns):
        pairs = X_train[["Department", "JobRole"]].dropna().drop_duplicates()
        for dept, group in pairs.groupby("Department"):
            department_roles[dept] = sorted(group["JobRole"].tolist())

    bundle = {
        "pipeline": pipeline,                       # preprocessing + classifier together
        "feature_columns": X.columns.tolist(),      # exact column order used for training
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "numeric_info": numeric_info,
        "categories": categories,
        "department_roles": department_roles,
        "target": TARGET,
        "class_labels": {0: "Stay", 1: "Leave"},
        "metrics": metrics,
        "model_name": "Logistic Regression (tuned: C=0.1)",
        "sklearn_version": sklearn.__version__,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model bundle to: {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"Trained with scikit-learn {sklearn.__version__}")


if __name__ == "__main__":
    main()
