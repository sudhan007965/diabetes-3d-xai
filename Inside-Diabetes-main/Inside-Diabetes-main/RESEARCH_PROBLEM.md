# RESEARCH_PROBLEM.md

## Project Context
The **Inside Diabetes** project provides a 3‑D interactive visualisation of human anatomy alongside a client‑side AI model that predicts diabetes risk using the Pima Indians dataset. The current system uses a single Multilayer Perceptron (MLP) with raw sigmoid probabilities and a proxy “SHAP” waterfall.

## Research Goal
Upgrade the system into a **research‑oriented, reliable, explainable AI platform** that:
1. Benchmarks multiple classification algorithms on the same dataset.
2. Provides **well‑calibrated probability estimates**.
3. Supplies **true SHAP explanations** for individual predictions.
4. Enables **counterfactual exploration** through a side‑panel UI.
5. Improves overall code modularity while preserving the cinematic 3‑D visual identity.

## Selected Evaluation Models (Phase 3)
- **Logistic Regression** – a linear baseline that is easy to calibrate and interpret.
- *(Optional future extensions can add Random Forest, Gradient Boosting, SVM – but initial focus is on Logistic Regression as per user preference.)*

## Calibration Strategy (Phase 4)
- **Isotonic regression** via `sklearn.calibration.CalibratedClassifierCV` (recommended). This non‑parametric method preserves monotonicity and works well with limited data.
- The calibrated model will be exported alongside the existing MLP weights for front‑end use.

## Counterfactual Explorer UI
- **Layout**: a **side panel** attached to the existing input form. The panel will list each feature with a slider/input that allows the user to modify **one feature at a time** while keeping the others fixed, instantly updating the risk score and displaying the delta.
- The panel will also show the reliability badge (e.g., “Calibrated”) and a brief textual explanation of the change.

## Deliverables for Upcoming Phases
| Phase | Deliverable | Brief Description |
|-------|-------------|-------------------|
| **3** | `evaluate_models.py` | Script that trains Logistic Regression (and optionally other models), performs stratified 5‑fold cross‑validation, computes a full metric suite, and saves the best calibrated model to `model/calibrated_model.pkl` and a JSON payload for the front‑end. |
| **4** | Updated `model_weights.json` (or a new `calibrated_model.json`) containing calibrated probabilities and true SHAP values for a sample of test instances. |
| **4** | UI mock‑up (HTML/CSS) for the side‑panel counterfactual explorer – placed in `website/counterfactual.html` (or integrated into `index.html`). |
| **5‑6** | Refactored `script.js` modules (model inference, UI logic, 3‑D scene) – planned but not yet implemented. |

*The above plan aligns with the user‑selected models, calibration method, and UI layout.*
