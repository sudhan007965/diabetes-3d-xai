import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Load dataset
DATA_PATH = Path('c:/ANN/Diabetes_AI_Project/data/diabetes.csv')
if not DATA_PATH.is_file():
    raise FileNotFoundError(f'Dataset not found at {DATA_PATH}')

df = pd.read_csv(DATA_PATH, header=None)
if df.shape[1] == 9:
    df.columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']

# Replace zeros with NaN for specific columns
zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Impute missing values with medians
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler for front‑end usage
joblib.dump(scaler, Path('c:/ANN/Diabetes_AI_Project/model/scaler.pkl'))

# Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=200)
rf.fit(X_train_scaled, y_train)
joblib.dump(rf, Path('c:/ANN/Diabetes_AI_Project/model/rf_model.pkl'))
print('Random Forest saved')

# Gradient Boosting
gb = GradientBoostingClassifier(random_state=42, n_estimators=200)
gb.fit(X_train_scaled, y_train)
joblib.dump(gb, Path('c:/ANN/Diabetes_AI_Project/model/gb_model.pkl'))
print('Gradient Boosting saved')

# SVM with isotonic calibration
svm = SVC(probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)
svm_cal = CalibratedClassifierCV(svm, method='isotonic', cv=5)
svm_cal.fit(X_train_scaled, y_train)
joblib.dump(svm_cal, Path('c:/ANN/Diabetes_AI_Project/model/svm_model.pkl'))
print('Calibrated SVM saved')
