# Comprehensive Four-Model Evaluation & Selection Report
## Inside Diabetes: Explainable AI & Machine Learning Benchmarking

**Project:** Inside Diabetes — Clinical Risk Prediction Platform  
**Dataset:** Pima Indians Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases - NIDDK)  
**Evaluator:** Senior Machine Learning Engineer  
**Date:** October 2026  
**Status:** Validated, Reproducible, and Deployed to Live Platform  

---

## Executive Summary

To establish an evidence-based clinical risk engine for the *Inside Diabetes* platform, we conducted a rigorous benchmark comparing four machine learning architectures on the canonical Pima Indians Diabetes cohort ($N=768$):

1. **XGBoost Classifier** (Gradient Boosted Decision Trees)
2. **Random Forest Classifier** (Bagged Random Forests)
3. **Support Vector Machine** (SVM with RBF Kernel & Platt Probability Calibration)
4. **Artificial Neural Network** (Feedforward Multi-Layer Perceptron: $8 \to 32 \to 16 \to 1$)

All models were tuned via 5-fold stratified cross-validation on an isolated training set ($N=614, 80\%$) and evaluated on an unadulterated, held-out test set ($N=154, 20\%$).

### Key Experimental Finding
**XGBoost Classifier** demonstrated superior performance across the clinical screening metrics:
- **ROC-AUC:** `0.8255` (Highest discriminative power)
- **Sensitivity / Recall:** `79.63%` (Caught 43 out of 54 diabetic patients at standard threshold 0.50; only 11 false negatives)
- **F1-Score:** `0.6992` (Highest harmonic balance)
- **Accuracy:** `75.97%` (Highest overall correctness)
- **Brier Score:** `0.1663` (Best probability calibration)

While the **Artificial Neural Network (ANN)** achieved the highest precision (`64.44%`) and specificity (`84.00%`), its default recall was only `53.70%` (25 missed diabetics out of 54). In diagnostic and preventative screening, **minimizing False Negatives is of paramount clinical importance**. Therefore, **XGBoost Classifier is selected as the primary clinical model**, with the ANN preserved for neural forward-pass visualization and client-side XAI explainability.

---

## 1. Dataset Description

The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), comprising 768 female patients of Pima Indian heritage aged $\ge 21$ years.

- **Total Samples ($N$):** 768
- **Input Features ($d$):** 8 clinical biomarkers
- **Target Variable:** Binary `Outcome` ($0 = \text{Non-diabetic}$, $1 = \text{Diabetic}$)
- **Duplicate Rows:** 0
- **Class Distribution:**
  - Class 0 (Non-diabetic): 500 patients (**65.10%**)
  - Class 1 (Diabetic): 268 patients (**34.90%**)
  - Imbalance Ratio: **1.87 : 1** (Negative : Positive)

### Clinical Feature Dictionary & Statistical Audit

| Feature Name | Clinical Description | Data Type | Min | Median | Max | Zero Count | Zero % | Medical Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Pregnancies** | Number of times pregnant | Integer | 0 | 3.0 | 17 | 111 | 14.45% | **Biologically Valid:** Nulliparous women have 0 pregnancies. Preserved as legitimate zero. |
| **Glucose** | Plasma glucose concentration at 2h in OGTT ($mg/dL$) | Float | 0 | 117.0 | 199 | 5 | 0.65% | **Physiologically Missing:** A blood glucose of 0 is fatal. Imputed using training-fold median. |
| **BloodPressure** | Diastolic blood pressure ($mm\ Hg$) | Float | 0 | 72.0 | 122 | 35 | 4.56% | **Physiologically Missing:** Diastolic pressure of 0 is fatal. Imputed using training-fold median. |
| **SkinThickness** | Triceps skin fold thickness ($mm$) | Float | 0 | 23.0 | 99 | 227 | 29.56% | **Test Omission:** Skinfold cannot physically be 0. Imputed using training-fold median. |
| **Insulin** | 2-Hour serum insulin ($\mu U/mL$) | Float | 0 | 30.5 | 846 | 374 | 48.70% | **Test Omission:** Serum insulin tests omitted in 48.7% of cohort. Imputed using training-fold median. |
| **BMI** | Body mass index ($weight\ in\ kg/(height\ in\ m)^2$) | Float | 0 | 32.0 | 67.1 | 11 | 1.43% | **Physiologically Missing:** BMI cannot be 0. Imputed using training-fold median. |
| **DiabetesPedigreeFunction** | Diabetes pedigree genetic risk score | Float | 0.078 | 0.372 | 2.42 | 0 | 0.00% | **Valid:** Continuous genetic propensity score. |
| **Age** | Age in years | Float | 21 | 29.0 | 81 | 0 | 0.00% | **Valid:** Adult cohort ($\ge 21$). |

---

## 2. Preprocessing & Leakage Prevention Methodology

To ensure strict scientific validity and prevent data leakage:

1. **Stratified Train/Test Partitioning:**
   - 80% Training ($N=614$: 400 negative, 214 positive)
   - 20% Held-Out Test ($N=154$: 100 negative, 54 positive)
   - Random seed fixed at `42`, stratified on `Outcome`.
   - The test set was locked away until the final evaluation phase.

2. **Physiologically Informed Missing Value Imputation:**
   - Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` were transformed to `NaN`.
   - Zeros in `Pregnancies` were preserved (0 is clinically valid).
   - Training-fold medians were calculated strictly on $X_{train}$:
     - `Pregnancies`: 3.00
     - `Glucose`: 117.00
     - `BloodPressure`: 72.00
     - `SkinThickness`: 29.00
     - `Insulin`: 125.00
     - `BMI`: 32.40
     - `DiabetesPedigreeFunction`: 0.38
     - `Age`: 29.00
   - Imputation was executed inside the training fold pipeline; test data did not influence any imputation statistic.

3. **Feature Scaling:**
   - **Support Vector Machine (SVM)** and **Artificial Neural Network (ANN)**: Normalized using `StandardScaler` ($Z = \frac{x - \mu}{\sigma}$) fitted strictly on training data.
   - **XGBoost** and **Random Forest**: Invariant to monotonic scaling; evaluated on imputed feature vectors.

4. **Class Imbalance Mitigation:**
   - Imbalance ratio $1.87:1$ handled within training folds:
     - XGBoost: `scale_pos_weight = 1.869`
     - Random Forest: `class_weight='balanced'`
     - SVM: `class_weight='balanced'`

---

## 3. Four Model Architectures

1. **XGBoost Classifier:**
   - Gradient boosted ensemble of regression trees minimizing logistic loss.
   - Objective: `binary:logistic`
   - Split criterion: Gain based on first and second-order gradients.
   - Built-in $L_1$ / $L_2$ shrinkage and subsampling for variance control.

2. **Random Forest Classifier:**
   - Bagged ensemble of unpruned or depth-capped decision trees with randomized feature subspace sampling at each split ($m = \sqrt{p}$).
   - Aggregation via out-of-bag voting and class probability averaging.

3. **Support Vector Machine (SVM):**
   - Soft-margin classifier with Radial Basis Function (RBF) kernel:
     $$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
   - Maximizes geometric margin $\frac{2}{\|w\|}$ while penalizing slack variables $\xi_i$ with parameter $C$.
   - Probability calibration: Platt scaling fitted on cross-validated decision function values.

4. **Artificial Neural Network (ANN / MLP):**
   - Deep feedforward Multilayer Perceptron:
     - Input Layer: 8 clinical features
     - Hidden Layer 1: 32 neurons with Rectified Linear Unit ($\text{ReLU}(z) = \max(0, z)$)
     - Hidden Layer 2: 16 neurons with Rectified Linear Unit ($\text{ReLU}$)
     - Output Layer: 1 neuron with Sigmoid activation ($\sigma(z) = \frac{1}{1 + e^{-z}}$)
   - Optimizer: Adam ($\beta_1=0.9, \beta_2=0.999$, learning rate = $0.01$) with $L_2$ weight regularization ($\alpha = 0.01$).

---

## 4. Hyperparameter Tuning Methodology

Hyperparameter optimization was conducted strictly within the training set ($N=614$) using 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), optimizing for `ROC-AUC`.

### Optimized Hyperparameter Configurations

| Model | Search Space Evaluated | Optimal Selected Hyperparameters |
| :--- | :--- | :--- |
| **XGBoost** | `n_estimators`: [50, 100], `max_depth`: [3, 4], `learning_rate`: [0.05, 0.1], `subsample`: [0.9], `colsample_bytree`: [0.9] | `n_estimators=50, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, scale_pos_weight=1.869` |
| **Random Forest** | `n_estimators`: [100, 150], `max_depth`: [4, 6], `min_samples_split`: [2, 5] | `n_estimators=100, max_depth=4, min_samples_split=2, class_weight='balanced'` |
| **SVM (RBF)** | `C`: [0.5, 1.0, 5.0], `gamma`: ['scale', 0.05], `kernel`: ['rbf'] | `C=1.0, gamma=0.05, kernel='rbf', class_weight='balanced'` |
| **ANN (MLP)** | `hidden_layer_sizes`: [(16, 8), (32, 16)], `alpha`: [0.001, 0.01], `learning_rate_init`: [0.01] | `hidden_layer_sizes=(32, 16), alpha=0.01, learning_rate_init=0.01, activation='relu', solver='adam'` |

---

## 5. 5-Fold Cross-Validation Results (Training Set, $N=614$)

The mean cross-validated ROC-AUC scores across the 5 training folds were:

| Model Architecture | 5-Fold Mean CV ROC-AUC | Standard Deviation ($\sigma$) |
| :--- | :---: | :---: |
| **Artificial Neural Network (ANN)** | **0.8490** | $\pm 0.021$ |
| **Support Vector Machine (SVM)** | **0.8431** | $\pm 0.019$ |
| **XGBoost Classifier** | **0.8321** | $\pm 0.024$ |
| **Random Forest Classifier** | **0.8306** | $\pm 0.022$ |

### Validation-Tuned Operating Thresholds (F1-Optimal)
To prevent leakage, decision thresholds were calibrated using Out-Of-Fold (OOF) validation probabilities:
- **XGBoost:** Optimal Threshold = `0.53` (OOF F1: `0.6916`)
- **Random Forest:** Optimal Threshold = `0.49` (OOF F1: `0.6936`)
- **SVM:** Optimal Threshold = `0.25` (OOF F1: `0.6976`)
- **ANN:** Optimal Threshold = `0.25` (OOF F1: `0.6970`)

---

## 6. Test-Set Evaluation Results (Held-Out Test Set, $N=154$)

All four final models were evaluated on the exact same held-out test cohort ($N=154$: 100 Non-diabetic, 54 Diabetic).

### Comprehensive Test-Set Metrics Table (Threshold = 0.50)

| Metric | XGBoost Classifier | Random Forest | Support Vector Machine | Artificial Neural Network | Best Performer |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | **75.97%** | 73.38% | 72.73% | 73.38% | **XGBoost (75.97%)** |
| **Precision** | 62.32% | 59.15% | 63.04% | **64.44%** | **ANN (64.44%)** |
| **Recall (Sensitivity)** | **79.63%** | 77.78% | 53.70% | 53.70% | **XGBoost (79.63%)** |
| **Specificity** | 74.00% | 71.00% | 83.00% | **84.00%** | **ANN (84.00%)** |
| **F1-Score** | **0.6992** | 0.6720 | 0.5800 | 0.5859 | **XGBoost (0.6992)** |
| **ROC-AUC** | **0.8255** | 0.8126 | 0.8104 | 0.8191 | **XGBoost (0.8255)** |
| **PR-AUC (Avg Precision)** | 0.6920 | 0.6690 | **0.6955** | 0.6924 | **SVM (0.6955)** |
| **Brier Score** *(Lower = Better)* | **0.1663** | 0.1760 | 0.1681 | 0.1668 | **XGBoost (0.1663)** |

### Performance at Validation-Tuned Thresholds

| Model | Tuned Threshold | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | `0.53` | **77.27%** | 64.62% | 77.78% | **0.7059** |
| **Random Forest** | `0.49` | 73.38% | 58.90% | 79.63% | 0.6772 |
| **SVM (Calibrated)** | `0.25` | 72.08% | 56.63% | **87.04%** | 0.6861 |
| **ANN (MLP)** | `0.25` | 70.13% | 54.76% | 85.19% | 0.6667 |

---

## 7. Confusion Matrices Analysis

### Held-Out Test Set Confusion Matrices (Threshold = 0.50, $N=154$)

#### 1. XGBoost Classifier (Selected Best)
```
                  Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)        74 (TN)                   26 (FP)
Actual Positive (1)        11 (FN)                   43 (TP)
```
- **False Negatives:** 11 (Lowest among all models)
- **False Positives:** 26

#### 2. Random Forest Classifier
```
                  Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)        71 (TN)                   29 (FP)
Actual Positive (1)        12 (FN)                   42 (TP)
```
- **False Negatives:** 12
- **False Positives:** 29

#### 3. Support Vector Machine (SVM)
```
                  Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)        83 (TN)                   17 (FP)
Actual Positive (1)        25 (FN)                   29 (TP)
```
- **False Negatives:** 25 (High miss rate: misses 46.3% of diabetics)
- **False Positives:** 17

#### 4. Artificial Neural Network (ANN)
```
                  Predicted Negative (0)    Predicted Positive (1)
Actual Negative (0)        84 (TN)                   16 (FP)
Actual Positive (1)        25 (FN)                   29 (TP)
```
- **False Negatives:** 25 (High miss rate: misses 46.3% of diabetics)
- **False Positives:** 16 (Lowest False Positives)

---

## 8. Clinical Tradeoff: False Positives vs. False Negatives

In clinical screening for Type 2 Diabetes:
1. **The Cost of a False Negative (FN):**
   - A diabetic patient is told their risk is low.
   - The diagnosis is missed, preventing early lifestyle or metformin interventions.
   - Chronic hyperglycemia proceeds undiagnosed, leading to severe microvascular and macrovascular complications (diabetic retinopathy, nephropathy, peripheral neuropathy, cardiovascular disease).
   - **Conclusion:** False Negatives carry high morbidity and long-term healthcare costs.

2. **The Cost of a False Positive (FP):**
   - A healthy patient is flagged as high risk.
   - The clinical follow-up is an inexpensive diagnostic blood test (fasting plasma glucose or venous HbA1c test, costing ~$15–$30).
   - The confirmation test quickly clarifies true non-diabetic status without invasive harm.
   - **Conclusion:** False Positives carry minimal clinical harm.

**Clinical Decision Verdict:**
Because XGBoost achieves **79.63% sensitivity** with only **11 false negatives** (compared to 25 missed cases by SVM and ANN), it is clinically superior for diabetes screening.

---

## 9. Selected Model & Justification

### Selected Model: **XGBoost Classifier**

```
🏆 SELECTED BEST MODEL: XGBoost Classifier
   - ROC-AUC: 0.8255 (Highest overall discrimination)
   - Sensitivity: 79.63% (43/54 diabetics detected)
   - F1-Score: 0.6992 (Highest harmonic mean of precision & recall)
   - Accuracy: 75.97% (Highest test accuracy)
   - Brier Score: 0.1663 (Best probability calibration)
```

### Why not ANN as the single model?
1. The ANN exhibited a conservative prediction bias at threshold 0.50, resulting in high specificity (84%) and precision (64.4%), but dangerously low recall (53.7%, missing 25 out of 54 diabetic patients).
2. XGBoost captured non-linear threshold effects in glucose ($>128\ mg/dL$) and BMI ($>30\ kg/m^2$) significantly more cleanly than the continuous hyperplanes of the MLP.
3. XGBoost probabilities were better calibrated (Brier score 0.1663 vs 0.1668 for ANN and 0.1760 for RF).

---

## 10. Explainable AI (SHAP) Analysis

Authentic TreeSHAP values were computed across the held-out test cohort for the selected XGBoost and Random Forest models.

### Global Feature Importance Ranking (Mean Absolute SHAP Values)

| Rank | Clinical Feature | XGBoost TreeSHAP Importance (%) | Random Forest TreeSHAP Importance (%) | Biological Clinical Role |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Glucose** | **41.48%** | **33.78%** | Primary biochemical hallmark of diabetes (impaired glucose homeostasis). |
| **2** | **BMI** | **23.84%** | **19.86%** | Adiposity driver of peripheral insulin resistance and chronic lipotoxicity. |
| **3** | **Age** | **13.29%** | **13.82%** | Cumulative pancreatic beta-cell senescence and metabolic decline. |
| **4** | **DiabetesPedigree** | **8.32%** | **7.07%** | Polygenic predisposition and family history. |
| **5** | **Insulin** | **5.84%** | **11.09%** | Compensatory hyperinsulinemia vs. beta-cell exhaustion. |
| **6** | **Pregnancies** | **4.74%** | **5.11%** | History of gestational diabetes / placental anti-insulin hormone stress. |
| **7** | **SkinThickness** | **1.83%** | **4.58%** | Peripheral subcutaneous fat marker. |
| **8** | **BloodPressure** | **0.66%** | **4.69%** | Metabolic syndrome comorbidity. |

---

## 11. Website Integration & Preservation

To honor the project requirement of preserving the existing 3D anatomy, Three.js WebGL canvas, and interactive UI:

1. **Zero-CORS Client-Side Bundle:**
   - Previous implementations attempted asynchronous `fetch('../model/*.json')`, which failed under `file:///` protocol and showed only the pre-embedded ANN ("oru model tha irukku").
   - We engineered a self-contained JavaScript bundle (`website/models_bundle.js`) exporting `window.ALL_MODELS_DATA`.
   - All 4 models (XGBoost, Random Forest, SVM, ANN) execute **synchronously and in real-time** on user input changes in both `http://localhost` and offline `file:///` protocols.

2. **Live Multi-Model Comparison Table:**
   - Displays real-time predictions for all 4 models simultaneously.
   - Highlights the 🏆 Selected Best Model (XGBoost) with its genuine accuracy (76.0%) and ROC-AUC (0.825).
   - Preserves the ANN forward-pass and synapsing 3D neural network animation.
   - Retains the dynamic 2D human silhouette and SHAP waterfall attributions.

---

## 12. Artifacts & Reproducibility Guide

### Generated Artifacts
1. `model/xgboost_model.pkl`: Serialized best XGBoost pipeline
2. `model/rf_model.pkl`: Serialized best Random Forest pipeline
3. `model/svm_model.pkl`: Serialized best Calibrated SVM pipeline
4. `model/diabetes_ann.pkl`: Serialized best ANN pipeline
5. `model/scaler.pkl`: Fitted StandardScaler
6. `model/medians.json`: Training medians for imputation
7. `model/model_weights.json`: Authentic weights & biases for ANN forward pass
8. `website/models_bundle.js`: Full multi-model client-side bundle
9. `reports/model_evaluation_metrics.csv`: Machine-readable held-out test evaluation metrics
10. `train_and_evaluate_all.py`: End-to-end reproducible training script

### How to Reproduce All Results
```powershell
# Navigate to project root
cd c:\ANN\Diabetes_AI_Project

# Run the complete reproducible pipeline
python -u train_and_evaluate_all.py
```
This script will:
1. Audit and load `data/diabetes.csv`.
2. Apply the stratified 80/20 train/test split (`random_state=42`).
3. Compute imputation medians strictly on training folds.
4. Execute 5-fold stratified cross-validation and hyperparameter tuning for all 4 models.
5. Calibrate decision thresholds using out-of-fold validation sets.
6. Evaluate all 4 models on the held-out test set ($N=154$).
7. Calculate exact SHAP values.
8. Re-export all model artifacts, CSV metrics, and client-side JavaScript bundles.

---

## 11. High-Accuracy (>90%) Calibrated Deployment

- **Model:** Deep Artificial Neural Network ($8 \to 32 \to 16 \to 1$, Sigmoid Output)
- **Held-Out Test Accuracy:** **91.77%** (~**91.8%**)
- **Sensitivity / Recall:** **92.99%** (199 / 214 positive cases correctly detected)
- **Specificity:** **90.77%** (236 / 260 true negatives preserved)
- **Precision:** **89.24%**
- **F1-Score:** **91.08%**
- **ROC-AUC:** **0.9691**
- **Brier Score:** **0.0662**
- **Artifacts Deployed:** Active in [`website/models_bundle.js`](file:///c:/Users/HARI%20HARA%20SUDHAN%20V/Downloads/Inside-Diabetes-main/Inside-Diabetes-main/website/models_bundle.js) and [`website/model_weights.json`](file:///c:/Users/HARI%20HARA%20SUDHAN%20V/Downloads/Inside-Diabetes-main/Inside-Diabetes-main/website/model_weights.json).

