# Inside Diabetes: Empirical Validation, Imputation Comparison & Uncertainty Analysis

**Senior Machine Learning & Clinical Decision Review**  
**Dataset:** Pima Indians Diabetes Database ($N=768$, 8 clinical biomarkers + 1 binary target)  
**Evaluation Cohort:** Strictly partitioned held-out test split ($N=154$, 20% stratified: 100 negatives, 54 positives; $N_{\text{train}}=614$, 80% stratified; `random_state=42`)

---

## 1. Executive Summary & Honest Statistical Reality

A rigorous empirical evaluation of the four clinical architectures (**XGBoost**, **Random Forest**, **Support Vector Machine**, and **Artificial Neural Network**) along with a **Soft-Voting Ensemble** and **MICE Imputation** was conducted. All imputers and scalers were fitted exclusively on training data (within cross-validation folds) to guarantee zero data leakage.

### Exact Parity with Original Held-Out Benchmarks:
Across all experiments, the saved production models reproduce the metrics in [reports/model_evaluation_metrics.csv](file:///c:/ANN/Diabetes_AI_Project/reports/model_evaluation_metrics.csv) exactly at default threshold 0.5:
- **XGBoost**: 117/154 correct (75.97% Acc, 0.8255 ROC-AUC, 0.1663 Brier)
- **Random Forest**: 113/154 correct (73.38% Acc, 0.8126 ROC-AUC, 0.1760 Brier)
- **Support Vector Machine**: 112/154 correct (72.73% Acc, 0.8104 ROC-AUC, 0.1681 Brier)
- **Artificial Neural Network**: 113/154 correct (73.38% Acc, 0.8191 ROC-AUC, 0.1668 Brier)

### Key Takeaways:
1. **Root Cause of Initial Metric Discrepancy:** `data/diabetes.csv` does not possess a column header line. When parsed with standard `pd.read_csv()`, row 0 was consumed as header names, shrinking the dataset from 768 to 767 rows and shifting the `train_test_split(random_state=42)` indices. Loading with `header=None` restores all 768 rows, perfectly aligning the test partition and reproducing all held-out numbers to 6 decimal places.
2. **Median vs. MICE Imputation:** Using the production models' tuned hyperparameters, Median Imputation produces identical numbers to original results. IterativeImputer (MICE) achieves 0.8155 test AUC for XGBoost, 0.8078 for RF, 0.8100 for SVM, and 0.8139 for ANN. While MICE slightly shifts feature relationships, it does not demonstrate statistically significant improvements over training-median imputation on held-out test samples.
3. **Uncertainty & Confidence Stratification:** Epistemic uncertainty (inter-model standard deviation $\sigma$) demonstrates an accuracy drop from **84.2% [69.6%, 92.6%]** in Q1 (most certain) down to **64.1% [48.4%, 77.3%]** in Q4 (most uncertain). Comparing against baseline confidence ($|P_{\text{ens}} - 0.5|$) shows an even steeper gradient: **97.4% [86.5%, 99.5%]** in Q1 (furthest from boundary) down to **43.6% [29.3%, 59.0%]** in Q4 (closest to 0.5), which is well below the **65.1% majority-class baseline**.
4. **Honest Statistical Assessment (McNemar & Bootstrap):** Using exact binomial testing for small discordance counts ($b + c < 25$), pairwise tests against XGBoost yield $p=0.2891$ for RF, $p=0.4414$ for SVM, $p=0.5708$ for ANN, and $p=0.3018$ for the Ensemble. With $N=154$, differences between models correspond to only 4–5 patients ($b - c \in [4, 5]$). All 95% bootstrap confidence intervals overlap substantially (spanning ~0.74 to ~0.89). No single model is statistically significantly superior to any other at $\alpha=0.05$.

---

## 2. Model Evaluation Parity Check (Threshold 0.5)

Comparison of model outputs on held-out test set ($N=154$) against [reports/model_evaluation_metrics.csv](file:///c:/ANN/Diabetes_AI_Project/reports/model_evaluation_metrics.csv):

| Model | Correct Predictions | Held-out Accuracy | ROC-AUC | Brier Score | Match Status |
|---|---|---|---|---|---|
| **XGBoost** | **117 / 154** (TN:74, TP:43) | **75.97%** | **0.8255** | **0.1663** | **Exact Match** |
| **Random Forest** | **113 / 154** (TN:71, TP:42) | **73.38%** | **0.8126** | **0.1760** | **Exact Match** |
| **Support Vector Machine** | **112 / 154** (TN:83, TP:29) | **72.73%** | **0.8104** | **0.1681** | **Exact Match** |
| **Artificial Neural Network** | **113 / 154** (TN:84, TP:29) | **73.38%** | **0.8191** | **0.1668** | **Exact Match** |
| **Soft-Voting Ensemble** | **112 / 154** (TN:78, TP:34) | **72.73%** | **0.8215** | **0.1638** | New Ensemble Benchmark |
| **Majority-Class Baseline** | **100 / 154** (All Negative) | **64.94%** (65.10% overall) | 0.5000 | 0.3506 | Naive Baseline |

---

## 3. Tuned Hyperparameters Audit

Tuned hyperparameters identified from grid search on training cross-validation and saved in `model/*.pkl`:

1. **XGBoost** (`xgboost_model.pkl`):
   - `n_estimators`: 50
   - `max_depth`: 3
   - `learning_rate`: 0.05
   - `subsample`: 0.9
   - `colsample_bytree`: 0.9
   - `scale_pos_weight`: 1.8692 ($N_{\text{neg}} / N_{\text{pos}} = 400 / 214$)
2. **Random Forest** (`rf_model.pkl`):
   - `n_estimators`: 100
   - `max_depth`: 4
   - `min_samples_split`: 5
   - `class_weight`: `'balanced'`
3. **Support Vector Machine** (`svm_model.pkl`):
   - `C`: 1.0
   - `gamma`: 0.05
   - `kernel`: `'rbf'`
   - `class_weight`: `'balanced'`
   - `probability`: True (Platt scaling: $A = -1.3237, B = -0.5699$)
4. **Artificial Neural Network** (`diabetes_ann.pkl`):
   - Architecture: 8 $\to$ 32 $\to$ 16 $\to$ 1
   - `activation`: `'relu'`
   - `solver`: `'adam'`
   - `alpha` (L2 penalty): 0.01
   - `learning_rate_init`: 0.01
   - `early_stopping`: True
   - `validation_fraction`: 0.15
   - `n_iter_no_change`: 25
   - `max_iter`: 1000

---

## 4. Imputation Benchmark: Median vs. IterativeImputer (MICE)

*Results saved in:* [reports/imputation_comparison.csv](file:///c:/ANN/Diabetes_AI_Project/reports/imputation_comparison.csv)

| Imputation Strategy | Model | Correct Count | Held-out Accuracy | Sensitivity | Test ROC-AUC | 5-Fold CV AUC |
|---|---|---|---|---|---|---|
| **Median Imputation** | XGBoost | 117 / 154 | 75.97% | 79.63% | 0.8255 | 0.8314 |
| **Median Imputation** | Random Forest | 113 / 154 | 73.38% | 77.78% | 0.8126 | 0.8384 |
| **Median Imputation** | Support Vector Machine | 112 / 154 | 72.73% | 53.70% | 0.8104 | 0.8430 |
| **Median Imputation** | Artificial Neural Network | 113 / 154 | 73.38% | 53.70% | 0.8191 | 0.8454 |
| **IterativeImputer (MICE)** | XGBoost | 116 / 154 | 75.32% | 75.93% | 0.8155 | 0.8338 |
| **IterativeImputer (MICE)** | Random Forest | 113 / 154 | 73.38% | 77.78% | 0.8078 | 0.8340 |
| **IterativeImputer (MICE)** | Support Vector Machine | 115 / 154 | 74.68% | 61.11% | 0.8100 | 0.8419 |
| **IterativeImputer (MICE)** | Artificial Neural Network | 111 / 154 | 72.08% | 55.56% | 0.8139 | 0.8411 |

---

## 5. Statistical Rigor: McNemar Paired Tests & Bootstrap 95% CIs

### A. McNemar Discordance Test vs. XGBoost (Exact Binomial Test for $b + c < 25$)
*Results saved in:* [reports/mcnemar_results.csv](file:///c:/ANN/Diabetes_AI_Project/reports/mcnemar_results.csv)

Note that mathematically, $b - c = \text{XGBoost Correct} - \text{Comparator Correct}$.

| Comparison | XGB Corr | Comp Corr | $b$ (XGB+ M-) | $c$ (XGB- M+) | $b - c$ | Statistical Test | $p$-value | Significant ($\alpha=0.05$)? |
|---|---|---|---|---|---|---|---|---|
| **XGBoost vs. Random Forest** | 117 | 113 | 6 | 2 | 4 | **Exact Binomial** | **0.2891** | **No ($p \ge 0.05$)** |
| **XGBoost vs. Support Vector Machine** | 117 | 112 | 16 | 11 | 5 | Chi-Square (cc) | **0.4414** | **No ($p \ge 0.05$)** |
| **XGBoost vs. Artificial Neural Network** | 117 | 113 | 16 | 12 | 4 | Chi-Square (cc) | **0.5708** | **No ($p \ge 0.05$)** |
| **XGBoost vs. Soft-Voting Ensemble** | 117 | 112 | 10 | 5 | 5 | **Exact Binomial** | **0.3018** | **No ($p \ge 0.05$)** |

### B. 1,000-Resample Stratified Bootstrap 95% Confidence Intervals for ROC-AUC
*Results saved in:* [reports/bootstrap_auc_ci.csv](file:///c:/ANN/Diabetes_AI_Project/reports/bootstrap_auc_ci.csv)

| Model | Test ROC-AUC | Bootstrap Mean | 95% CI Lower | 95% CI Upper | Standard Error |
|---|---|---|---|---|---|
| **XGBoost** | 0.8255 | 0.8258 | 0.7485 | 0.8879 | 0.0345 |
| **Random Forest** | 0.8126 | 0.8130 | 0.7341 | 0.8736 | 0.0356 |
| **Support Vector Machine** | 0.8104 | 0.8105 | 0.7348 | 0.8743 | 0.0357 |
| **Artificial Neural Network** | 0.8191 | 0.8189 | 0.7434 | 0.8820 | 0.0351 |
| **Soft-Voting Ensemble** | 0.8215 | 0.8216 | 0.7460 | 0.8839 | 0.0347 |

---

## 6. Uncertainty & Confidence Quartile Stratification

*Results saved in:* [reports/ensemble_results.csv](file:///c:/ANN/Diabetes_AI_Project/reports/ensemble_results.csv)  
*Figure saved in:* [reports/figures/uncertainty.png](file:///c:/ANN/Diabetes_AI_Project/reports/figures/uncertainty.png)

### Stratification by Epistemic Uncertainty (Model Disagreement $\sigma$):
| Quartile | $N$ | Correct | Accuracy | 95% Wilson CI | Mean $\sigma$ | Brier Score |
|---|---|---|---|---|---|---|
| **Q1 (Most Certain / High Consensus)** | 38 | 32 / 38 | **84.2%** | [69.6%, 92.6%] | 0.0309 | 0.1121 |
| **Q2 (Moderately Certain)** | 39 | 31 / 39 | **79.5%** | [64.5%, 89.2%] | 0.0511 | 0.1577 |
| **Q3 (Moderately Uncertain)** | 38 | 24 / 38 | **63.2%** | [47.3%, 76.6%] | 0.0720 | 0.1868 |
| **Q4 (Most Uncertain / High Disagreement)**| 39 | 25 / 39 | **64.1%** | [48.4%, 77.3%] | 0.1112 | 0.1978 |

### Stratification by Baseline Confidence ($|P_{\text{ens}} - 0.5|$):
| Quartile | $N$ | Correct | Accuracy | 95% Wilson CI | Mean Confidence | Brier Score |
|---|---|---|---|---|---|---|
| **Q1 (Highest Confidence)** | 38 | 37 / 38 | **97.4%** | [86.5%, 99.5%] | 0.4198 | 0.0294 |
| **Q2 (Moderately High Confidence)**| 39 | 30 / 39 | **76.9%** | [61.7%, 87.4%] | 0.3244 | 0.1754 |
| **Q3 (Moderately Low Confidence)** | 38 | 28 / 38 | **73.7%** | [58.0%, 85.0%] | 0.1954 | 0.1914 |
| **Q4 (Lowest Confidence / Boundary)**| 39 | 17 / 39 | **43.6%** | [29.3%, 59.0%] | 0.0623 | 0.2562 |

### Key Insight:
- While epistemic uncertainty ($\sigma$) captures genuine inter-model divergence, baseline confidence $|P_{\text{ens}} - 0.5|$ captures proximity to the decision boundary.
- In Q1 of confidence, the ensemble achieves **97.4% accuracy**, making it extremely reliable for clinical triage.
- In Q4 of confidence, accuracy drops to **43.6%**, which is substantially worse than the **65.1% naive majority-class baseline**, proving that patients within $\pm 0.06$ of the 0.5 threshold should never receive automated diagnoses and must undergo confirmatory lab testing.

---

## 7. Calibration Curves & Decision Curve Analysis

- **Expected Calibration Error (ECE):**
  - Soft-Voting Ensemble: **0.0786** (Brier: 0.1638) — lowest calibration error
  - ANN: 0.0921 (Brier: 0.1668)
  - Random Forest: 0.0944 (Brier: 0.1760)
  - XGBoost: 0.0972 (Brier: 0.1663)
  - SVM: 0.0973 (Brier: 0.1681)
  - *Figure saved in:* [reports/figures/calibration_curves.png](file:///c:/ANN/Diabetes_AI_Project/reports/figures/calibration_curves.png)
- **Decision Curve Analysis (Net Clinical Benefit):**
  - Across screening threshold probabilities $p_t \in [0.15, 0.60]$, all machine learning models yield strictly positive net benefit superior to both "Screen All" and "Screen None".
  - *Figure saved in:* [reports/figures/decision_curve_analysis.png](file:///c:/ANN/Diabetes_AI_Project/reports/figures/decision_curve_analysis.png)
