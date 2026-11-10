import json, joblib, os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# Load scaler used for all models (created during training)
scaler_path = os.path.join('model', 'scaler.pkl')
if not os.path.exists(scaler_path):
    raise FileNotFoundError('Scaler file not found: ' + scaler_path)
scaler = joblib.load(scaler_path)
mean = scaler.mean_.tolist()
std = scaler.scale_.tolist()

def tree_to_dict(tree):
    def recurse(node):
        if tree.children_left[node] == -1:  # leaf node
            # tree.value shape: (1, 1, n_classes) – average probability for positive class
            return {"value": float(tree.value[node].mean())}
        return {
            "feature": int(tree.feature[node]),
            "threshold": float(tree.threshold[node]),
            "left": recurse(tree.children_left[node]),
            "right": recurse(tree.children_right[node])
        }
    return recurse(0)

def export(model, name, out_path):
    # Determine tree structures
    if hasattr(model, 'estimators_'):
        # RandomForest stores a list; GradientBoosting stores a 2‑D ndarray
        if isinstance(model.estimators_, list):
            trees = [tree_to_dict(est.tree_) for est in model.estimators_]
        else:
            # GradientBoosting: estimator at [0, 0]
            trees = [tree_to_dict(model.estimators_[0, 0].tree_)]
    else:
        trees = []
    out = {
        "model_name": name,
        "metrics": {"accuracy": getattr(model, 'best_score_', None)},
        "scaler_mean": mean,
        "scaler_std": std,
        "trees": trees
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Exported {out_path}')

# Load trained models (paths are from evaluate_models.py)
rf = joblib.load(os.path.join('model', 'rf_model.pkl'))
gb = joblib.load(os.path.join('model', 'gb_model.pkl'))
svm = joblib.load(os.path.join('model', 'svm_model.pkl'))

export(rf, "RandomForest", os.path.join('model', 'archive', 'random_forest_calibrated.json'))
export(gb, "GradientBoosting", os.path.join('model', 'gradient_boosting_calibrated.json'))

# Export SVM calibration details (CalibratedClassifierCV wrapper)
svm_out = {
    "model_name": "SVM",
    "metrics": {"accuracy": getattr(svm, 'best_score_', None)},
    "scaler_mean": mean,
    "scaler_std": std,
    "coefficients": (getattr(svm, 'base_estimator_', None) and getattr(svm.base_estimator_, 'coef_', None) and svm.base_estimator_.coef_[0].tolist()) or [],
    "intercept": (getattr(svm, 'base_estimator_', None) and getattr(svm.base_estimator_, 'intercept_', None) and svm.base_estimator_.intercept_.tolist()) or [],
    "platt_A": (svm.calibrated_classifiers_ and len(svm.calibrated_classifiers_) > 0 and getattr(svm.calibrated_classifiers_[0], 'probA_', None)) or None,
    "platt_B": (svm.calibrated_classifiers_ and len(svm.calibrated_classifiers_) > 0 and getattr(svm.calibrated_classifiers_[0], 'probB_', None)) or None
}
with open(os.path.join('model', 'archive', 'svm_calibrated.json'), 'w') as f:
    json.dump(svm_out, f, indent=2)
print('Exported svm_calibrated.json to archive')
