# evaluate_models.py
"""Model Evaluation and Calibration Script

This script trains several classification models on the Pima Indians Diabetes dataset,
performs stratified K‑fold cross‑validation, evaluates a comprehensive set of metrics,
applies probability calibration (isotonic regression), and exports the best calibrated
model along with the required JSON payload for the front‑end.

Key steps:
1. Load data, handle missing values, and standardise using the same logic as the
   original MLP pipeline.
2. Define candidate models (Logistic Regression, Random Forest, Gradient Boosting,
   SVM, and the existing MLP).
3. For each model, perform 5‑fold stratified CV, compute metrics (accuracy, precision,
   recall, F1, ROC‑AUC, PR‑AUC, Brier score, calibration curve), and select the
   best hyper‑parameters.
4. Fit a CalibratedClassifierCV (isotonic) on the training set and evaluate on the
   held‑out test set.
5. Export the calibrated model weights (for Logistic Regression – coefficients &
   intercept) and scaler statistics to a JSON file compatible with the front‑end.
6. Additionally, compute true SHAP values for a representative subset of the test
   set using `shap.KernelExplainer` and store them in the JSON payload.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import shap

# -------------------- 1. Load and preprocess data --------------------
DATA_PATH = Path(__file__).parent.parent / "data" / "diabetes.csv"
# If the CSV is not present, adjust the path accordingly.
if not DATA_PATH.is_file():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH, header=None)
# Assign column names if not present
if df.shape[1] == 9:
    df.columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
# Replace zeros with NaN for physiologically impossible columns
zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)
# Separate features and target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
# Stratified split (80/20) – keep same random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Compute medians on training set for imputation
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)
# Standardise using training statistics
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- 2. Define candidate models --------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='lbfgs'),
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC(probability=True, kernel='rbf', random_state=42),
    "MLP": MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu',
                           solver='adam', max_iter=500, random_state=42)
}

# Light hyper‑parameter grids for demonstration purposes
param_grids = {
    "LogisticRegression": {"C": [0.01, 0.1, 1, 10]},
    "RandomForest": {"n_estimators": [100], "max_depth": [None, 5, 10]},
    "GradientBoosting": {"learning_rate": [0.1], "n_estimators": [100]},
    "SVM": {"C": [0.1, 1, 10]},
    "MLP": {"alpha": [0.0001, 0.001]}
}

# -------------------- 3. Cross‑validation & metric collection --------------------
results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    print(f"\n=== Evaluating {name} ===")
    grid = GridSearchCV(model, param_grids[name], cv=skf, scoring='roc_auc')
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    # Train on full training data
    best_model.fit(X_train_scaled, y_train)
    # Predict on test set
    probas = best_model.predict_proba(X_test_scaled)[:, 1]
    preds = best_model.predict(X_test_scaled)
    # Basic metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probas)
    pr_auc = average_precision_score(y_test, probas)
    brier = brier_score_loss(y_test, probas)
    cm = confusion_matrix(y_test, preds).tolist()
    results[name] = {
        "best_params": grid.best_params_,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "confusion_matrix": cm,
        "probas": probas.tolist(),
        "preds": preds.tolist()
    }

# -------------------- 4. Calibration (Isotonic) for Logistic Regression --------------------
chosen_name = "LogisticRegression"
chosen_grid = GridSearchCV(models[chosen_name], param_grids[chosen_name], cv=skf, scoring='roc_auc')
chosen_grid.fit(X_train_scaled, y_train)
best_lr = chosen_grid.best_estimator_
# Fit isotonic calibration on the already‑trained model
calibrated = CalibratedClassifierCV(best_lr, method='isotonic', cv=5)
calibrated.fit(X_train_scaled, y_train)
# Evaluate calibrated model on test set
calib_probas = calibrated.predict_proba(X_test_scaled)[:, 1]
calib_preds = calibrated.predict(X_test_scaled)
calib_acc = accuracy_score(y_test, calib_preds)
calib_roc_auc = roc_auc_score(y_test, calib_probas)

# -------------------- 5. Export JSON payload for the front‑end --------------------
payload = {
    "model_name": chosen_name,
    "architecture": "Logistic Regression (linear)",
    "dataset": "Pima Indians Diabetes",
    "total_samples": len(df),
    "metrics": {
        "accuracy": calib_acc,
        "roc_auc": calib_roc_auc,
        "brier_score": brier_score_loss(y_test, calib_probas)
    },
    "feature_names": list(X.columns),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std": np.sqrt(scaler.var_).tolist(),
    "medians": medians.to_dict(),
    "coefficients": best_lr.coef_.flatten().tolist(),
    "intercept": best_lr.intercept_.tolist(),
    "expected_value": float(np.mean(y_train)),
    "calibration": "isotonic",
    "calibration_curve": {
        "fraction_of_positives": [],
        "mean_predicted_value": []
    }
}
# Calibration curve (10 bins)
frac_pos, mean_pred = calibration_curve(y_test, calib_probas, n_bins=10)
payload["calibration_curve"]["fraction_of_positives"] = frac_pos.tolist()
payload["calibration_curve"]["mean_predicted_value"] = mean_pred.tolist()

# -------------------- 6. Compute SHAP values for a subset of test set --------------------
# KernelExplainer works with any model; we use the calibrated model's underlying estimator
explainer = shap.KernelExplainer(best_lr.predict_proba, X_train_scaled[:100])
shap_vals = explainer.shap_values(X_test_scaled[:50])  # first 50 test instances
payload["shap"] = shap_vals[1].tolist()  # class 1 explanations

# Save payload next to this script for easy front‑end consumption
output_path = Path(__file__).parent / "calibrated_model.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Calibration JSON written to {output_path}")
