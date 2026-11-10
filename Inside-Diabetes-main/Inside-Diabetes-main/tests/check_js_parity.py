"""
Inside Diabetes — Python vs. JavaScript Pure ANN Parity Verifier
Compares predictions from trained Python ANN model (diabetes_ann.pkl) with the
client-side JavaScript bundle across 20 diverse clinical test inputs.
"""

import os
import json
import math
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
WEBSITE_DIR = os.path.join(BASE_DIR, 'website')

def run_parity_check():
    print("=" * 80)
    print("INSIDE DIABETES: PYTHON VS JAVASCRIPT PREDICTION PARITY CHECK (PURE ANN - 20 INPUTS)")
    print("=" * 80)

    # 1. Load Python model & scaler
    ann_model = joblib.load(os.path.join(MODEL_DIR, 'diabetes_ann.pkl'))
    scaler    = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

    with open(os.path.join(MODEL_DIR, 'medians.json'), 'r') as f:
        medians = json.load(f)

    # 2. Load JavaScript bundle
    with open(os.path.join(WEBSITE_DIR, 'models_bundle.js'), 'r', encoding='utf-8') as f:
        js_text = f.read()
    json_str = js_text.split('window.ALL_MODELS_DATA = ')[1].split(';\n')[0]
    bundle = json.loads(json_str)

    feature_names = bundle['metadata']['feature_names']

    # 3. JS Emulation Function for ANN
    def evaluate_js_ann(raw_vec):
        b_meta = bundle['metadata']
        b_medians = b_meta['medians']
        b_mean = b_meta['scaler_mean']
        b_std = b_meta['scaler_std']

        imputed = []
        for i, name in enumerate(feature_names):
            v = raw_vec[i]
            if name in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and (v == 0 or np.isnan(v)):
                imputed.append(b_medians[name])
            else:
                imputed.append(v)

        scaled = [(imputed[i] - b_mean[i]) / b_std[i] for i in range(8)]
        m = bundle['models'][0]

        W1, W2, W3 = m['weights']
        b1, b2, b3 = m['biases']
        h1 = [max(0.0, b1[j] + sum(scaled[i] * W1[i][j] for i in range(8))) for j in range(len(b1))]
        h2 = [max(0.0, b2[k] + sum(h1[j] * W2[j][k] for j in range(len(h1)))) for k in range(len(b2))]
        logit = b3[0] + sum(h2[k] * W3[k][0] for k in range(len(h2)))
        return 1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, logit))))

    # 4. Generate 20 diverse test inputs
    test_cases = [
        ("Healthy Young",       [1, 85, 68, 18, 65, 21.8, 0.22, 25]),
        ("Borderline Normal",   [2, 110, 74, 22, 90, 25.4, 0.31, 32]),
        ("Impaired Fasting",    [2, 122, 78, 28, 135, 28.6, 0.45, 44]),
        ("High Metabolic Risk", [4, 178, 88, 38, 210, 36.2, 0.88, 52]),
        ("Severe Diabetic",     [8, 197, 92, 45, 320, 42.5, 1.25, 59]),
        ("Dataset Medians",     [3, 117, 72, 29, 125, 32.4, 0.38, 29]),
        ("Physiological Zeros", [2, 0, 0, 0, 0, 0, 0.38, 33]),
        ("Nullipara Young",     [0, 92, 64, 20, 72, 22.1, 0.18, 21]),
        ("Nullipara Older",     [0, 140, 80, 32, 160, 31.0, 0.52, 54]),
        ("High Pedigree Risk",  [3, 130, 76, 26, 140, 29.5, 2.15, 38]),
        ("Low BMI High Glucose",[1, 168, 70, 16, 110, 20.5, 0.35, 41]),
        ("High BMI Low Glucose",[3, 88, 84, 40, 95, 41.2, 0.28, 36]),
        ("Elderly Patient",     [5, 135, 82, 30, 115, 27.8, 0.42, 72]),
        ("Young Overweight",    [1, 105, 68, 24, 80, 33.1, 0.29, 23]),
        ("Gestational Profile", [6, 145, 80, 34, 185, 34.0, 0.62, 35]),
        ("High Blood Pressure", [3, 125, 104, 32, 120, 30.5, 0.48, 50]),
        ("Low Insulin Profile", [1, 115, 72, 22, 35, 24.8, 0.25, 28]),
        ("Metabolic Boundary",  [2, 126, 75, 27, 130, 30.0, 0.40, 40]),
        ("High Risk Multipara", [10, 160, 86, 36, 190, 35.5, 0.75, 48]),
        ("Athletic Low Risk",   [0, 80, 62, 14, 55, 19.8, 0.15, 22])
    ]

    max_diff = 0.0

    print(f"\n{'Test Case':22s} | {'Py Prob':9s} | {'JS Prob':9s} | {'Delta':10s}")
    print("-" * 55)

    for case_name, raw in test_cases:
        x_raw = np.array([raw], dtype=float)
        x_py = x_raw.copy()
        for i, name in enumerate(feature_names):
            if name in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] and x_py[0, i] == 0:
                x_py[0, i] = medians[name]
        x_py_scaled = scaler.transform(x_py)

        py_prob = float(ann_model.predict_proba(x_py_scaled)[0, 1])
        js_prob = evaluate_js_ann(raw)

        diff = abs(py_prob - js_prob)
        if diff > max_diff:
            max_diff = diff

        print(f"{case_name:22s} | {py_prob:9.4f} | {js_prob:9.4f} | {diff:10.6f}")

    print("-" * 55)
    print(f"Maximum absolute delta across 20 inputs: {max_diff:.8f}")
    assert max_diff < 1e-6, "Parity check failed: discrepancy exceeds 1e-6!"
    print("SUCCESS: Python and JavaScript Pure ANN predictions match perfectly!")
    print("=" * 80)

if __name__ == '__main__':
    run_parity_check()
