import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1) LOAD DATA
df = pd.read_excel("AllPatients1.xlsx")  # file must be next to this script

# 2) CREATE TARGET: Disease = 0/1
df["Disease"] = df["Diagnosis"].apply(lambda x: 0 if x == "No Cardiac Problem" else 1)

# 3) KEEP ONLY THE COLUMNS WE USE IN THE UI
selected_cols = [
    "HeartRate", "O2Saturation", "ECG", "Echo", "Troponin",
    "ECGIschemicChanges", "ChestPainType", "BNP",
    "Hypertension", "STChanges", "WallMotionAbnormalities",
    "Disease"
]
df = df[selected_cols]

# 4) ENCODE ANY TEXT COLUMNS (if there are any)
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col].astype(str))

# 5) SPLIT FEATURES / TARGET
X = df.drop(columns=["Disease"])
y = df["Disease"]

# 6) SCALE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7) TRAIN RANDOM FOREST
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 8) QUICK CHECK AUC
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Random Forest ROC-AUC: {auc:.3f}")

# 9) SAVE MODEL + SCALER
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved model.pkl and scaler.pkl")
