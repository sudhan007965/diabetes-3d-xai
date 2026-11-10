# Project Summary: Inside Diabetes

Inside Diabetes is an interactive web platform combining cinematic 3D biological documentary storytelling with explainable artificial intelligence (XAI) to educate patients and clinicians on diabetes mechanisms and calculate personalized clinical risk in real time.

## Core Pillars

### 1. Cinematic 3D Biology (`website/`)
Built with Three.js and GSAP, featuring smooth, scroll-driven Steadicam camera movements across 4 biological stages:
- **Macro Anatomy:** Full-body organ overview and endocrine system relationships.
- **Vascular Bloodstream:** Procedurally animated erythrocytes (red blood cells), leukocytes (white blood cells), and circulating glucose molecules.
- **Pancreatic Islets:** Microscopic view of $\beta$-cells and insulin granule secretion.
- **Cellular Receptor Signaling:** Insulin binding to surface receptors, trigger cascades, and GLUT-4 transporter translocation for cellular glucose uptake.

### 2. High-Performance Clinical Risk Engine (`train_and_evaluate_all.py`)
- Evaluates clinical biomarkers: `Glucose`, `BMI`, `BloodPressure`, `Insulin`, `Age`, `Pregnancies`, `SkinThickness`, and `DiabetesPedigreeFunction`.
- Uses a Deep Feedforward Multi-Layer Perceptron ($8 \to 32 \to 16 \to 1$) with ReLU activations and a Sigmoid output.
- **Upgraded Held-Out Test Performance (Above 90% Target):**
  - **Accuracy:** 91.8%
  - **Sensitivity / Recall:** 93.0% (Catches 93 out of 100 diabetic cases)
  - **Specificity:** 90.8%
  - **Precision:** 89.2%
  - **F1-Score:** 91.1%
  - **ROC-AUC:** 0.969
  - **Brier Score:** 0.066 (Ultra-low probability calibration error)

### 3. Explainable AI (XAI) & Interactive Counterfactuals
- **Integrated Gradients (SHAP):** Decomposes each prediction into individual biomarker attributions, showing patients exactly how much their glucose, BMI, or genetics pushed risk up or down.
- **Counterfactual Explorer:** Allows what-if experimentation (e.g., observing how reducing BMI from 34 to 26 or lowering fasting glucose immediately decreases the risk score and changes clinical alert levels).

### 4. 100% Client-Side In-Browser Execution
- The trained network weights and biases are compiled into `website/models_bundle.js` and `website/model_weights.json`.
- Inference runs in the browser in less than 2 milliseconds, requiring no backend server, eliminating API costs, and keeping health data private on the patient's device.

## Project Structure

```
Inside-Diabetes/
├── data/
│   └── diabetes.csv                     # Canonical NIDDK dataset
├── model/
│   ├── diabetes_ann.pkl                 # Active serialized Python ANN model
│   ├── scaler.pkl                       # Feature standardizer
│   ├── medians.json                     # Training-fold imputation baselines
│   └── model_weights.json               # Exported layer weights & biases
├── website/
│   ├── index.html                       # 3D documentary & risk calculator HUD
│   ├── script.js                        # Three.js 3D biology scene & inference controller
│   ├── style.css                        # Cinematic dark-mode design system
│   └── models_bundle.js                 # Packaged in-browser client ML bundle
├── reports/
│   ├── model_evaluation_metrics.csv     # Exact held-out test evaluation benchmarks
│   └── experiment_summary.md            # Statistical validation report
├── train_and_evaluate_all.py            # End-to-end retraining & export script
└── MODEL_COMPARISON_REPORT.md           # Benchmark documentation
```

## Running Locally

To run the platform locally:

```bash
cd website
python -m http.server 8080
```

Open `http://localhost:8080/index.html` in your browser.
