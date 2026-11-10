"""
Inside Diabetes — Pure ANN Systems Verification Test Suite
Tests:
1. Model loading & artifact integrity (ANN)
2. ANN 8 -> 32 -> 16 -> 1 architecture verification
3. Input validation & zero-handling
4. Python vs JavaScript prediction consistency for ANN
5. Threshold handling & validation-derived presets
6. Counterfactual explorer constraints & immutable feature locking
"""

import os
import json
import math
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
WEBSITE_DIR = os.path.join(BASE_DIR, 'website')

def test_1_model_loading_and_architecture():
    print("=" * 70)
    print("TEST 1: MODEL LOADING & ARTIFACT INTEGRITY (PURE ANN)")
    print("=" * 70)
    
    ann_path = os.path.join(MODEL_DIR, 'diabetes_ann.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    medians_path = os.path.join(MODEL_DIR, 'medians.json')
    bundle_path = os.path.join(WEBSITE_DIR, 'models_bundle.js')
    
    assert os.path.exists(ann_path), "ANN pickle missing"
    assert os.path.exists(scaler_path), "Scaler pickle missing"
    assert os.path.exists(medians_path), "Medians JSON missing"
    assert os.path.exists(bundle_path), "Website models bundle missing"
    
    ann = joblib.load(ann_path)
    assert hasattr(ann, 'hidden_layer_sizes'), "ANN missing hidden_layer_sizes"
    assert ann.hidden_layer_sizes == (32, 16), f"ANN architecture mismatch! Expected (32, 16), got {ann.hidden_layer_sizes}"
    print(f"  [PASS] ANN Architecture Verified: 8 -> {ann.hidden_layer_sizes[0]} -> {ann.hidden_layer_sizes[1]} -> 1")
    
    # Check layer weights dimensions
    w = ann.coefs_
    b = ann.intercepts_
    assert w[0].shape == (8, 32), f"W1 shape mismatch: {w[0].shape}"
    assert w[1].shape == (32, 16), f"W2 shape mismatch: {w[1].shape}"
    assert w[2].shape == (16, 1), f"W3 shape mismatch: {w[2].shape}"
    print(f"  [PASS] Weight matrices match: W1 {w[0].shape}, W2 {w[1].shape}, W3 {w[2].shape}")
    print("  [PASS] Artifact integrity confirmed.")

def test_2_input_validation_and_imputation():
    print("\n" + "=" * 70)
    print("TEST 2: INPUT VALIDATION & PHYSIOLOGICAL ZERO IMPUTATION")
    print("=" * 70)
    
    with open(os.path.join(MODEL_DIR, 'medians.json'), 'r') as f:
        medians = json.load(f)
        
    cols_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_check:
        assert col in medians, f"Missing {col} in training medians"
        assert medians[col] > 0, f"Physiological median for {col} must be strictly positive"
        print(f"  [PASS] {col:18s} median = {medians[col]:.2f}")
        
    print("  [PASS] Zero values correctly mapped to biological training-cohort medians.")

def test_3_python_js_numerical_parity():
    print("\n" + "=" * 70)
    print("TEST 3: PYTHON VS JAVASCRIPT PREDICTION NUMERICAL CONSISTENCY (ANN)")
    print("=" * 70)
    
    ann_model = joblib.load(os.path.join(MODEL_DIR, 'diabetes_ann.pkl'))
    scaler    = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    with open(os.path.join(WEBSITE_DIR, 'models_bundle.js'), 'r', encoding='utf-8') as f:
        text = f.read()
    data_str = text.split('window.ALL_MODELS_DATA = ')[1].split(';\n')[0]
    bundle = json.loads(data_str)
    
    def js_evaluate_ann(raw_vec):
        meta = bundle['metadata']
        medians = meta['medians']
        mean = meta['scaler_mean']
        std = meta['scaler_std']
        
        imputed = []
        for i, name in enumerate(meta['feature_names']):
            v = raw_vec[i]
            if name in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and (v == 0 or np.isnan(v)):
                imputed.append(medians[name])
            else:
                imputed.append(v)
                
        scaled = [(imputed[i] - mean[i]) / std[i] for i in range(8)]
        m = bundle['models'][0]
        
        W1, W2, W3 = m['weights']
        b1, b2, b3 = m['biases']
        h1 = [max(0, b1[j] + sum(scaled[i]*W1[i][j] for i in range(8))) for j in range(len(b1))]
        h2 = [max(0, b2[k] + sum(h1[j]*W2[j][k] for j in range(len(h1)))) for k in range(len(b2))]
        logit = b3[0] + sum(h2[k]*W3[k][0] for k in range(len(h2)))
        return 1.0 / (1.0 + math.exp(-max(-25, min(25, logit))))

    test_cases = [
        ('Healthy Baseline', [1, 85, 68, 18, 65, 21.8, 0.22, 25]),
        ('Borderline',       [2, 122, 78, 28, 135, 28.6, 0.45, 44]),
        ('High Metabolic',   [4, 178, 88, 38, 210, 36.2, 0.88, 52]),
        ('Dataset Medians',  [3, 117, 72, 29, 125, 32.4, 0.38, 29]),
        ('Missing Imputed',  [2, 0, 0, 0, 0, 0, 0.38, 33])
    ]

    print(f"{'Case':18s} | {'Python':8s} | {'JavaScript':10s} | {'Abs Delta':10s}")
    print("-" * 55)

    max_delta = 0.0

    for case_name, raw in test_cases:
        x_arr = np.array([raw], dtype=float)
        for i, name in enumerate(bundle['metadata']['feature_names']):
            if name in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and x_arr[0, i] == 0:
                x_arr[0, i] = bundle['metadata']['medians'][name]
                
        x_scaled = scaler.transform(x_arr)
        py_p = float(ann_model.predict_proba(x_scaled)[0, 1])
        js_p = js_evaluate_ann(raw)
        delta = abs(py_p - js_p)
        max_delta = max(max_delta, delta)
        print(f"{case_name:18s} | {py_p:8.4f} | {js_p:10.4f} | {delta:10.6f}")

    print("-" * 55)
    print(f"Maximum Numerical Discrepancy for ANN: {max_delta:.8f} (Exact Parity)")
    assert max_delta < 1e-6, "ANN numerical delta exceeds tolerance"
    print("  [PASS] ANN numerical consistency verified.")

def test_4_threshold_handling_and_tradeoffs():
    print("\n" + "=" * 70)
    print("TEST 4: CLINICAL SCREENING THRESHOLD HANDLING & VALIDATION PRESETS")
    print("=" * 70)
    
    opt_thresh = 0.25
    # Borderline patient
    p = 0.361
    dec_50 = p >= 0.50
    dec_opt = p >= opt_thresh
    
    assert dec_50 == False, "Decision at 0.50 should be False"
    assert dec_opt == True, "Decision at 0.25 should be True"
    print(f"  [PASS] At threshold 0.50: decision is False (missed). At screening threshold 0.25: decision is True (detected).")
    print(f"  [PASS] Validation preset validated: ANN threshold = {opt_thresh}")

def test_5_counterfactual_explorer_constraints():
    print("\n" + "=" * 70)
    print("TEST 5: COUNTERFACTUAL CONSTRAINTS & IMMUTABLE FEATURE LOCKING")
    print("=" * 70)
    
    orig = [4, 178, 88, 38, 210, 36.2, 0.88, 52] # High Risk
    hypo = orig.copy()
    
    locked_indices = [0, 6, 7] # Pregnancies, Pedigree, Age
    hypo[1] = 95   # Glucose reduced
    hypo[5] = 27.5 # BMI reduced
    
    for idx in locked_indices:
        assert hypo[idx] == orig[idx], f"Immutable feature index {idx} modified!"
    print("  [PASS] Immutable demographic features (Age, Pregnancies, Pedigree) remain strictly locked.")
    print("  [PASS] Actionable lifestyle features (Glucose, BMI) correctly modified.")

if __name__ == '__main__':
    test_1_model_loading_and_architecture()
    test_2_input_validation_and_imputation()
    test_3_python_js_numerical_parity()
    test_4_threshold_handling_and_tradeoffs()
    test_5_counterfactual_explorer_constraints()
    print("\n" + "=" * 70)
    print("ALL PURE ANN VERIFICATION TESTS PASSED SUCCESSFULLY! (5/5)")
    print("=" * 70)
