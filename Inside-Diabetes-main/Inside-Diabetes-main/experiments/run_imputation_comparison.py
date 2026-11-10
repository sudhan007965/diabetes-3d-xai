"""
Inside Diabetes — Experiment 1: Imputation Comparison (Median vs. IterativeImputer / MICE)
Compares standard median imputation against Multivariate Imputation by Chained Equations (MICE)
across all 4 models: XGBoost, Random Forest, SVM (RBF), and ANN (MLP 8->32->16->1).

Uses the saved models' exact tuned hyperparameters from train_and_evaluate_all.py:
- XGBoost: n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=1.869
- Random Forest: n_estimators=100, max_depth=4, min_samples_split=5, class_weight='balanced'
- SVM: C=1.0, gamma=0.05, kernel='rbf', class_weight='balanced', probability=True
- ANN: hidden_layer_sizes=(32, 16), alpha=0.01, learning_rate_init=0.01, activation='relu', solver='adam', early_stopping=True, validation_fraction=0.15, n_iter_no_change=25, max_iter=1000

Protocol:
- Preserves all 768 rows (handles missing header in diabetes.csv)
- Stratified 80/20 train/test split (random_state=42, N_train=614, N_test=154)
- 5-fold stratified cross-validation on training data only
- Zero leakage: Imputers and scalers fitted strictly inside CV folds and on training split
- Output saved to: reports/imputation_comparison.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
REPORT_PATH = os.path.join(BASE_DIR, 'reports', 'imputation_comparison.csv')
ORIG_METRICS_PATH = os.path.join(BASE_DIR, 'reports', 'model_evaluation_metrics.csv')

FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
COLS_WITH_ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def get_tuned_models(scale_pos_weight=1.8691588785046729):
    """Returns models instantiated with the exact tuned hyperparameters of the saved production models."""
    return {
        'XGBoost': XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        ),
        'Support Vector Machine': SVC(
            C=1.0,
            kernel='rbf',
            gamma=0.05,
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        'Artificial Neural Network': MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate_init=0.01,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            max_iter=1000,
            random_state=42
        )
    }

def print_hyperparameter_audit():
    print("=" * 85)
    print("MODEL HYPERPARAMETER AUDIT (EXACT TUNED PRODUCTION SPECIFICATION)")
    print("=" * 85)
    params_desc = {
        'XGBoost': {
            'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05,
            'subsample': 0.9, 'colsample_bytree': 0.9, 'scale_pos_weight': '1.8692 (imbalance ratio)'
        },
        'Random Forest': {
            'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 5, 'class_weight': 'balanced'
        },
        'Support Vector Machine': {
            'C': 1.0, 'gamma': 0.05, 'kernel': 'rbf', 'class_weight': 'balanced', 'probability': True
        },
        'Artificial Neural Network': {
            'hidden_layer_sizes': '(32, 16)', 'activation': 'relu', 'solver': 'adam',
            'alpha': 0.01, 'learning_rate_init': 0.01, 'early_stopping': True, 'n_iter_no_change': 25
        }
    }
    for m, p in params_desc.items():
        print(f"\n[{m}]")
        for k, v in p.items():
            print(f"  - {k:22s}: {v}")

def run_experiment():
    print_hyperparameter_audit()

    print("\n" + "=" * 85)
    print("EXPERIMENT 1: MEDIAN VS. ITERATIVE IMPUTER (MICE) BENCHMARK")
    print("=" * 85)

    # 1. Dataset Ingestion & Partitioning
    with open(DATA_PATH, 'r') as f:
        first_line = f.readline().strip()
    has_header = first_line.startswith('Pregnancies')
    df = pd.read_csv(DATA_PATH, header=0 if has_header else None)
    if df.shape[1] == 9:
        df.columns = FEATURE_NAMES + ['Outcome']

    X = df[FEATURE_NAMES].copy()
    y = df['Outcome'].values

    # Treat 0 as NaN in physiologically impossible biomarkers
    for col in COLS_WITH_ZERO_AS_MISSING:
        X[col] = X[col].replace(0, np.nan)

    # Stratified 80/20 train/test split (N_train=614, N_test=154)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    imputers = {
        'Median Imputation': lambda: SimpleImputer(strategy='median'),
        'IterativeImputer (MICE)': lambda: IterativeImputer(random_state=42, max_iter=20, sample_posterior=False)
    }

    results = []

    for imp_name, imp_factory in imputers.items():
        print(f"\n--- Evaluating Strategy: {imp_name} ---")
        
        # Fit imputer and scaler on full training split for held-out test evaluation
        imputer = imp_factory()
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp)
        X_test_scaled = scaler.transform(X_test_imp)

        base_models = get_tuned_models(scale_pos_weight=scale_pos)

        for model_name, model in base_models.items():
            use_scaling = (model_name in ['Support Vector Machine', 'Artificial Neural Network'])
            
            # Cross-Validation strictly fitting imputer & scaler on each fold to prevent leakage
            cv_aucs = []
            for tr_idx, val_idx in cv.split(X_train_raw, y_train):
                fold_imp = imp_factory()
                X_tr_fold = fold_imp.fit_transform(X_train_raw.iloc[tr_idx])
                X_val_fold = fold_imp.transform(X_train_raw.iloc[val_idx])

                if use_scaling:
                    fold_scaler = StandardScaler()
                    X_tr_fold = fold_scaler.fit_transform(X_tr_fold)
                    X_val_fold = fold_scaler.transform(X_val_fold)

                m_fold = get_tuned_models(scale_pos_weight=scale_pos)[model_name]
                m_fold.fit(X_tr_fold, y_train[tr_idx])
                val_probs = m_fold.predict_proba(X_val_fold)[:, 1]
                cv_aucs.append(roc_auc_score(y_train[val_idx], val_probs))

            # Train on full training split and evaluate on held-out test cohort
            X_tr_eval = X_train_scaled if use_scaling else X_train_imp
            X_te_eval = X_test_scaled if use_scaling else X_test_imp

            model.fit(X_tr_eval, y_train)
            test_probs = model.predict_proba(X_te_eval)[:, 1]
            test_preds = (test_probs >= 0.5).astype(int)

            n_corr = int(np.sum(test_preds == y_test))
            acc = float(accuracy_score(y_test, test_preds))
            sens = float(recall_score(y_test, test_preds))
            f1 = float(f1_score(y_test, test_preds))
            auc = float(roc_auc_score(y_test, test_probs))
            brier = float(brier_score_loss(y_test, test_probs))
            mean_cv_auc = float(np.mean(cv_aucs))

            print(f"  {model_name:26s} | Correct: {n_corr:3d}/154 | Acc: {acc*100:5.2f}% | Sens: {sens*100:5.2f}% | Test AUC: {auc:6.4f} | 5-Fold CV AUC: {mean_cv_auc:6.4f}")

            results.append({
                'Imputation Method': imp_name,
                'Model': model_name,
                'Correct Count': n_corr,
                'Held-out Accuracy': round(acc, 4),
                'Sensitivity': round(sens, 4),
                'F1-Score': round(f1, 4),
                'Held-out ROC-AUC': round(auc, 4),
                'Brier Score': round(brier, 4),
                '5-Fold Mean CV ROC-AUC': round(mean_cv_auc, 4)
            })

    # Save to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(REPORT_PATH, index=False)
    print(f"\n[SAVED] Imputation comparison results saved to {REPORT_PATH}")

    # Parity check against original results
    print("\n" + "=" * 85)
    print("PARITY VERIFICATION: MEDIAN IMPUTATION VS ORIGINAL HELD-OUT METRICS")
    print("=" * 85)
    df_orig = pd.read_csv(ORIG_METRICS_PATH).set_index('Model')
    median_subset = df_results[df_results['Imputation Method'] == 'Median Imputation'].set_index('Model')

    print(f"{'Model':26s} | {'Median Acc':10s} | {'Orig Acc':10s} | {'Median AUC':10s} | {'Orig AUC':10s} | {'Match?'}")
    print("-" * 85)
    for m in ['XGBoost', 'Random Forest', 'Support Vector Machine', 'Artificial Neural Network']:
        m_acc = median_subset.loc[m, 'Held-out Accuracy']
        o_acc = round(float(df_orig.loc[m, 'Accuracy']), 4)
        m_auc = median_subset.loc[m, 'Held-out ROC-AUC']
        o_auc = round(float(df_orig.loc[m, 'ROC-AUC']), 4)
        match_str = "EXACT MATCH" if (m_acc == o_acc and m_auc == o_auc) else "MISMATCH"
        print(f"{m:26s} | {m_acc*100:6.2f}%    | {o_acc*100:6.2f}%    | {m_auc:8.4f}   | {o_auc:8.4f}   | {match_str}")

    print("\n" + "=" * 85)
    print("EXPERIMENT 1 COMPLETED SUCCESSFULLY.")
    print("=" * 85)

if __name__ == '__main__':
    run_experiment()
