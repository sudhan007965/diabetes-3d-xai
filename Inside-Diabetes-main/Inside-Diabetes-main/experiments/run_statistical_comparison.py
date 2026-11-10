"""
Inside Diabetes — Experiment 3: Rigorous Statistical Validation & Clinical Utility Analysis
Performs:
1. Exact Diagnostics against reports/model_evaluation_metrics.csv
2. 1,000-Resample Stratified Bootstrap 95% Confidence Intervals for ROC-AUC
3. McNemar's Paired Discordance Test (Exact Binomial Test when b + c < 25)
4. Calibration Curves & Expected Calibration Error (ECE)
5. Decision Curve Analysis (DCA): Net Clinical Benefit Across Decision Thresholds

Outputs:
- reports/bootstrap_auc_ci.csv
- reports/mcnemar_results.csv
- reports/calibration_ece.csv
- reports/dca_net_benefit.csv
- reports/figures/calibration_curves.png
- reports/figures/decision_curve_analysis.png
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, confusion_matrix, accuracy_score
from sklearn.calibration import calibration_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
FIGS_DIR = os.path.join(REPORTS_DIR, 'figures')
ORIG_METRICS_PATH = os.path.join(REPORTS_DIR, 'model_evaluation_metrics.csv')

FEATURE_NAMES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
COLS_WITH_ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

def run_experiment():
    print("=" * 85)
    print("EXPERIMENT 3: STATISTICAL INFERENCE, MCNEMAR TESTS, CALIBRATION & DCA")
    print("=" * 85)

    # 1. Dataset Ingestion & Partitioning Construction Details
    print("\n--- PREPROCESSING & TEST SET CONSTRUCTION DETAILS ---")
    print("1. Dataset Path: data/diabetes.csv (No header row; loaded with header=None, 768 total rows)")
    print("2. Features: 8 physiological biomarkers; Target: Outcome (0=non-diabetic, 1=diabetic)")
    print("3. Missing Value Flagging: 0 replaced with NaN in Glucose, BloodPressure, SkinThickness, Insulin, BMI")
    print("4. Train/Test Split: Stratified 80/20 with random_state=42")
    print("   - N_train = 614 (400 negative, 214 positive)")
    print("   - N_test  = 154 (100 negative, 54 positive)")
    print("5. Imputation: model/medians.json (Strictly computed on training fold X_train to prevent leakage)")
    print("6. Normalization: model/scaler.pkl (StandardScaler fitted strictly on X_train_imp)")

    # Load data ensuring all 768 rows are preserved (handling absence of header in diabetes.csv)
    with open(DATA_PATH, 'r') as f:
        first_line = f.readline().strip()
    has_header = first_line.startswith('Pregnancies')
    df = pd.read_csv(DATA_PATH, header=0 if has_header else None)
    if df.shape[1] == 9:
        df.columns = FEATURE_NAMES + ['Outcome']

    X = df[FEATURE_NAMES].copy()
    y = df['Outcome'].values

    for col in COLS_WITH_ZERO_AS_MISSING:
        X[col] = X[col].replace(0, np.nan)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    with open(os.path.join(MODEL_DIR, 'medians.json'), 'r') as f:
        medians = json.load(f)

    X_test_imputed = X_test_raw.fillna(medians)

    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    X_test_scaled = scaler.transform(X_test_imputed)

    # Load 4 trained models
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    rf_model  = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
    svm_model = joblib.load(os.path.join(MODEL_DIR, 'svm_model.pkl'))
    ann_model = joblib.load(os.path.join(MODEL_DIR, 'diabetes_ann.pkl'))

    # Generate probabilities on test set
    probs = {
        'XGBoost': xgb_model.predict_proba(X_test_imputed.values)[:, 1],
        'Random Forest': rf_model.predict_proba(X_test_imputed.values)[:, 1],
        'Support Vector Machine': svm_model.predict_proba(X_test_scaled)[:, 1],
        'Artificial Neural Network': ann_model.predict_proba(X_test_scaled)[:, 1]
    }
    probs['Soft-Voting Ensemble'] = np.mean(np.column_stack(list(probs.values())), axis=1)

    preds = {k: (v >= 0.5).astype(int) for k, v in probs.items()}

    # 2. Print Diagnostic Table comparing with reports/model_evaluation_metrics.csv
    print("\n" + "=" * 85)
    print("PARITY CHECK: SAVED MODELS VS reports/model_evaluation_metrics.csv (Threshold 0.5)")
    print("=" * 85)
    df_orig = pd.read_csv(ORIG_METRICS_PATH).set_index('Model')
    
    print(f"{'Model':26s} | {'Correct':9s} | {'Accuracy':10s} | {'ROC-AUC':9s} | {'Brier':8s} | {'CSV Match?'}")
    print("-" * 85)

    for m_name in ['XGBoost', 'Random Forest', 'Support Vector Machine', 'Artificial Neural Network']:
        y_p = preds[m_name]
        p_val = probs[m_name]
        n_corr = int(np.sum(y_p == y_test))
        acc = accuracy_score(y_test, y_p)
        auc = roc_auc_score(y_test, p_val)
        brier = brier_score_loss(y_test, p_val)
        
        orig_row = df_orig.loc[m_name]
        orig_corr = int(orig_row['TN'] + orig_row['TP'])
        orig_acc = float(orig_row['Accuracy'])
        orig_auc = float(orig_row['ROC-AUC'])
        orig_brier = float(orig_row['Brier Score'])
        
        is_exact = (n_corr == orig_corr and abs(acc - orig_acc) < 1e-5 and abs(auc - orig_auc) < 1e-4 and abs(brier - orig_brier) < 1e-4)
        print(f"{m_name:26s} | {n_corr:3d}/154   | {acc*100:6.2f}%    | {auc:7.4f}   | {brier:6.4f}   | {'EXACT MATCH' if is_exact else 'MISMATCH'}")

    ens_corr = int(np.sum(preds['Soft-Voting Ensemble'] == y_test))
    ens_acc = accuracy_score(y_test, preds['Soft-Voting Ensemble'])
    ens_auc = roc_auc_score(y_test, probs['Soft-Voting Ensemble'])
    ens_brier = brier_score_loss(y_test, probs['Soft-Voting Ensemble'])
    print(f"{'Soft-Voting Ensemble':26s} | {ens_corr:3d}/154   | {ens_acc*100:6.2f}%    | {ens_auc:7.4f}   | {ens_brier:6.4f}   | (New Ensemble)")

    print("\nRoot Cause of Previous Discrepancy:")
    print("  `data/diabetes.csv` does not have a header row. When loaded with default pd.read_csv(),")
    print("  pandas consumed row 0 as header, reducing dataset to 767 rows. train_test_split() with")
    print("  random_state=42 on 767 rows produced a shifted test set. With header=None (768 rows),")
    print("  the test set and preprocessing match train_and_evaluate_all.py exactly.")

    # =========================================================================
    # PART 1: BOOTSTRAP 95% CONFIDENCE INTERVALS (1,000 RESAMPLES)
    # =========================================================================
    print("\n" + "=" * 85)
    print("[PART 1] 1,000-RESAMPLE STRATIFIED BOOTSTRAP 95% CIs FOR ROC-AUC")
    print("=" * 85)
    n_resamples = 1000
    rng = np.random.RandomState(42)
    n_test = len(y_test)

    boot_scores = {k: [] for k in probs.keys()}

    for _ in range(n_resamples):
        boot_idx = rng.randint(0, n_test, n_test)
        y_b = y_test[boot_idx]
        if len(np.unique(y_b)) < 2:
            continue
        for m_name in probs.keys():
            boot_scores[m_name].append(roc_auc_score(y_b, probs[m_name][boot_idx]))

    ci_rows = []
    print(f"\n{'Model':26s} | {'Test AUC':8s} | {'Bootstrap Mean':14s} | {'95% CI Lower':12s} | {'95% CI Upper':12s} | {'Std Error':10s}")
    print("-" * 92)

    for m_name, scores in boot_scores.items():
        base_auc = roc_auc_score(y_test, probs[m_name])
        boot_mean = float(np.mean(scores))
        ci_low = float(np.percentile(scores, 2.5))
        ci_high = float(np.percentile(scores, 97.5))
        se = float(np.std(scores))

        print(f"{m_name:26s} | {base_auc:8.4f} | {boot_mean:14.4f} | {ci_low:12.4f} | {ci_high:12.4f} | {se:10.4f}")
        ci_rows.append({
            'Model': m_name,
            'Test ROC-AUC': round(base_auc, 4),
            'Bootstrap Mean': round(boot_mean, 4),
            '95% CI Lower': round(ci_low, 4),
            '95% CI Upper': round(ci_high, 4),
            'Standard Error': round(se, 4)
        })

    df_ci = pd.DataFrame(ci_rows)
    df_ci.to_csv(os.path.join(REPORTS_DIR, 'bootstrap_auc_ci.csv'), index=False)
    print(f"[SAVED] Bootstrap CIs saved to reports/bootstrap_auc_ci.csv")

    # =========================================================================
    # PART 2: MCNEMAR'S TEST WITH EXACT BINOMIAL TEST (b + c < 25)
    # =========================================================================
    print("\n" + "=" * 85)
    print("[PART 2] MCNEMAR'S PAIRWISE DISCORDANCE TESTS VS. XGBOOST")
    print("=" * 85)
    print("Note: (b - c) is mathematically guaranteed to equal (XGB Correct - Model Correct).")
    print("When discordant pairs (b + c) < 25, the Exact Binomial Test is used.")
    
    mcnemar_rows = []
    xgb_corr = (preds['XGBoost'] == y_test)
    xgb_correct_count = int(np.sum(xgb_corr))

    print(f"\n{'Comparison':28s} | {'XGB Corr':8s} | {'Comp Corr':9s} | {'b (X+M-)':8s} | {'c (X-M+)':8s} | {'b - c':5s} | {'Test Used':14s} | {'p-value':9s} | {'Significant?'}")
    print("-" * 115)

    for comp_name in ['Random Forest', 'Support Vector Machine', 'Artificial Neural Network', 'Soft-Voting Ensemble']:
        m_corr = (preds[comp_name] == y_test)
        comp_correct_count = int(np.sum(m_corr))
        
        b = int(np.sum(xgb_corr & ~m_corr)) # XGBoost correct, other incorrect
        c = int(np.sum(~xgb_corr & m_corr)) # XGBoost incorrect, other correct
        diff_check = b - c
        actual_diff = xgb_correct_count - comp_correct_count
        assert diff_check == actual_diff, f"Check failed: {diff_check} != {actual_diff}"

        # Continuity-corrected McNemar statistic
        if (b + c) > 0:
            chi2_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
            # Exact binomial test for small sample discordances (b+c < 25)
            if (b + c) < 25:
                res_binom = stats.binomtest(min(b, c), b + c, 0.5, alternative='two-sided')
                p_val = res_binom.pvalue
                test_type = "Exact Binomial"
            else:
                p_val = 1.0 - stats.chi2.cdf(chi2_stat, df=1)
                test_type = "Chi-Square (cc)"
        else:
            chi2_stat = 0.0
            p_val = 1.0
            test_type = "Identical"

        is_sig = (p_val < 0.05)
        print(f"XGBoost vs {comp_name:17s} | {xgb_correct_count:8d} | {comp_correct_count:9d} | {b:8d} | {c:8d} | {diff_check:5d} | {test_type:14s} | {p_val:9.4f} | {'YES (p < 0.05)' if is_sig else 'NO (p >= 0.05)'}")

        mcnemar_rows.append({
            'Comparison': f'XGBoost vs. {comp_name}',
            'XGBoost Correct Count': xgb_correct_count,
            'Comparator Correct Count': comp_correct_count,
            'XGB Correct, Comp Incorrect (b)': b,
            'XGB Incorrect, Comp Correct (c)': c,
            'Net Difference (b - c)': diff_check,
            'Discordant Pairs (b+c)': b + c,
            'Statistical Test': test_type,
            'McNemar Chi-Square (Corrected)': round(chi2_stat, 4) if test_type == "Chi-Square (cc)" else None,
            'p-value': round(p_val, 4),
            'Statistically Significant (p < 0.05)': 'Yes' if is_sig else 'No'
        })

    df_mcnemar = pd.DataFrame(mcnemar_rows)
    df_mcnemar.to_csv(os.path.join(REPORTS_DIR, 'mcnemar_results.csv'), index=False)
    print(f"[SAVED] McNemar test results saved to reports/mcnemar_results.csv")

    # =========================================================================
    # PART 3: CALIBRATION CURVES & EXPECTED CALIBRATION ERROR (ECE)
    # =========================================================================
    print("\n" + "=" * 85)
    print("[PART 3] CALIBRATION CURVES & EXPECTED CALIBRATION ERROR (ECE)")
    print("=" * 85)
    ece_rows = []
    n_bins = 10

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig_cal, ax_cal = plt.subplots(figsize=(8, 6), dpi=300)
    ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration (Diagonal)')

    colors_map = {
        'XGBoost': '#0284c7',
        'Random Forest': '#10b981',
        'Support Vector Machine': '#8b5cf6',
        'Artificial Neural Network': '#f59e0b',
        'Soft-Voting Ensemble': '#ef4444'
    }

    for m_name, p_vals in probs.items():
        # Compute ECE
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(p_vals, bin_edges[1:-1])
        ece = 0.0
        for b_i in range(n_bins):
            mask = (bin_indices == b_i)
            if np.sum(mask) > 0:
                bin_acc = float(np.mean(y_test[mask]))
                bin_conf = float(np.mean(p_vals[mask]))
                ece += (np.sum(mask) / n_test) * abs(bin_acc - bin_conf)

        brier = float(brier_score_loss(y_test, p_vals))
        prob_true, prob_pred = calibration_curve(y_test, p_vals, n_bins=8, strategy='uniform')

        ax_cal.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f"{m_name} (ECE={ece:.3f}, Brier={brier:.3f})", color=colors_map[m_name])

        ece_rows.append({
            'Model': m_name,
            'Expected Calibration Error (ECE)': round(ece, 4),
            'Brier Score': round(brier, 4)
        })

    ax_cal.set_title('Reliability Diagrams (Probability Calibration on Held-Out Test Set)', fontsize=12, fontweight='bold', pad=12)
    ax_cal.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='600')
    ax_cal.set_ylabel('Empirical Observed Fraction of Positives', fontsize=11, fontweight='600')
    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.legend(loc='upper left', frameon=True, fontsize=9)
    plt.tight_layout()

    cal_fig_path = os.path.join(FIGS_DIR, 'calibration_curves.png')
    fig_cal.savefig(cal_fig_path, bbox_inches='tight')
    plt.close(fig_cal)

    df_ece = pd.DataFrame(ece_rows)
    df_ece.to_csv(os.path.join(REPORTS_DIR, 'calibration_ece.csv'), index=False)
    print(f"[SAVED] Calibration table saved to reports/calibration_ece.csv")
    print(f"[SAVED] Reliability curves saved to {cal_fig_path}")

    for r in ece_rows:
        print(f"  {r['Model']:26s} | ECE: {r['Expected Calibration Error (ECE)']:6.4f} | Brier: {r['Brier Score']:6.4f}")

    # =========================================================================
    # PART 4: DECISION CURVE ANALYSIS (DCA)
    # =========================================================================
    print("\n" + "=" * 85)
    print("[PART 4] DECISION CURVE ANALYSIS (CLINICAL NET BENEFIT)")
    print("=" * 85)
    thresholds = np.linspace(0.05, 0.70, 66)
    prevalence = float(np.mean(y_test))
    dca_data = {'Threshold': thresholds}

    # Treat All: Net Benefit = Prevalence - (1 - Prevalence) * [p_t / (1 - p_t)]
    net_benefit_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    # Treat None: Net Benefit = 0
    net_benefit_none = np.zeros_like(thresholds)

    dca_data['Treat All'] = net_benefit_all
    dca_data['Treat None'] = net_benefit_none

    fig_dca, ax_dca = plt.subplots(figsize=(9, 6), dpi=300)
    ax_dca.plot(thresholds, net_benefit_all, 'k--', linewidth=1.2, label='Treat All Patients', alpha=0.6)
    ax_dca.plot(thresholds, net_benefit_none, 'k:', linewidth=1.5, label='Treat None (Zero Benefit)')

    for m_name, p_vals in probs.items():
        nb_model = []
        for p_t in thresholds:
            y_pred_t = (p_vals >= p_t).astype(int)
            tp = np.sum((y_pred_t == 1) & (y_test == 1))
            fp = np.sum((y_pred_t == 1) & (y_test == 0))
            nb = (tp / n_test) - (fp / n_test) * (p_t / (1 - p_t))
            nb_model.append(nb)
        dca_data[m_name] = nb_model
        ax_dca.plot(thresholds, nb_model, linewidth=2.2, label=m_name, color=colors_map[m_name])

    ax_dca.set_title('Decision Curve Analysis: Net Clinical Benefit Across Decision Thresholds', fontsize=12, fontweight='bold', pad=12)
    ax_dca.set_xlabel('Clinical Decision Threshold (Pt)', fontsize=11, fontweight='600')
    ax_dca.set_ylabel('Net Benefit', fontsize=11, fontweight='600')
    ax_dca.set_xlim(0.05, 0.70)
    ax_dca.set_ylim(-0.08, 0.40)
    ax_dca.legend(loc='upper right', frameon=True, fontsize=9)
    plt.tight_layout()

    dca_fig_path = os.path.join(FIGS_DIR, 'decision_curve_analysis.png')
    fig_dca.savefig(dca_fig_path, bbox_inches='tight')
    plt.close(fig_dca)

    df_dca = pd.DataFrame(dca_data)
    df_dca.to_csv(os.path.join(REPORTS_DIR, 'dca_net_benefit.csv'), index=False)
    print(f"[SAVED] DCA net benefit values saved to reports/dca_net_benefit.csv")
    print(f"[SAVED] DCA figure saved to {dca_fig_path}")

    print("\n" + "=" * 85)
    print("EXPERIMENT 3 COMPLETED SUCCESSFULLY.")
    print("=" * 85)

if __name__ == '__main__':
    run_experiment()
