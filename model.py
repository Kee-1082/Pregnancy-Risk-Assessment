# model.py
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys

# --------------------------
# Config
# --------------------------
DATA_DIR = Path("data")

# Updated to include all columns in your dataset
EXPECTED_COLUMNS = [
    "age", "blood_pressure", "diastolic", "blood_sugar", "body_temp", 
    "bmi", "previous_complications", "preexisting_diabetes", 
    "gestational_diabetes", "mental_health", "heart_rate", "risk"
]

# Column synonyms mapping (common variants -> canonical name)
COLUMN_MAP = {
    "age": ["age", "age_years", "years"],
    "blood_pressure": ["blood_pressure", "bp", "bloodpressure", "blood pressure", "systolic_bp", "systolic"],
    "diastolic": ["diastolic", "diastolic_bp", "dbp", "diastolic blood pressure"],
    "blood_sugar": ["blood_sugar", "bloodsugar", "glucose", "sugar", "blood sugar", "bs"],
    "body_temp": ["body_temp", "temperature", "body_temperature", "temp"],
    "bmi": ["bmi", "body_mass_index", "body mass index"],
    "previous_complications": ["previous_complications", "complications", "prev_complications"],
    "preexisting_diabetes": ["preexisting_diabetes", "diabetes", "pre_diabetes"],
    "gestational_diabetes": ["gestational_diabetes", "gest_diabetes"],
    "mental_health": ["mental_health", "mental"],
    "heart_rate": ["heart_rate", "hr", "pulse", "heart rate", "heart_rate"],
    "risk": ["risk", "label", "outcome", "target", "risk_level"]
}

def canonicalize_columns(df):
    # lowercase, strip spaces
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    rename = {}
    cols = list(df.columns)
    for canon, variants in COLUMN_MAP.items():
        for v in variants:
            vnorm = v.strip().lower().replace(" ", "_")
            if vnorm in cols:
                rename[vnorm] = canon
                break
    if rename:
        df = df.rename(columns=rename)
    return df

# --------------------------
# Find files
# --------------------------
if not DATA_DIR.exists():
    print(f"❌ Data folder '{DATA_DIR}' does not exist. Create a folder named 'data' and put your files inside.", file=sys.stderr)
    sys.exit(1)

patterns = ["*.xlsx", "*.xls", "*.csv"]
files = []
for p in patterns:
    files.extend(list(DATA_DIR.glob(p)))

print(f"🔎 Found {len(files)} files: {[str(f.name) for f in files]}")

# --------------------------
# Read and combine
# --------------------------
df_list = []
for f in files:
    print(f"➡️  Reading {f.name} ...")
    try:
        if f.suffix.lower() in [".xlsx", ".xls"]:
            # try excel read (openpyxl/xlrd)
            df = pd.read_excel(f)  # pandas will choose engine; we installed openpyxl/xlrd
        elif f.suffix.lower() == ".csv":
            df = pd.read_csv(f)
        else:
            print(f"   ⚠️ Unsupported extension {f.suffix}, skipping.")
            continue
    except Exception as e:
        print(f"   ❌ Failed to read {f.name}: {e}")
        continue

    df = canonicalize_columns(df)
    print(f"   Columns after normalize: {list(df.columns)}")

    # Keep only columns we care about if present
    # We now only require age, blood_pressure, blood_sugar, and risk to be present
    required_columns = ["age", "blood_pressure", "blood_sugar", "risk"]
    present = [c for c in required_columns if c in df.columns]
    if len(present) < len(required_columns):
        print(f"   ⚠️ File {f.name} doesn't have required columns (found: {present}, need: {required_columns}), skipping.")
        continue

    # Keep only expected columns (others ignored)
    keep = [c for c in EXPECTED_COLUMNS if c in df.columns]
    df = df[keep]
    df_list.append(df)

# If nothing read, stop with helpful message
if not df_list:
    print("❌ No valid dataframes to concatenate. Check the 'data' folder and file columns.", file=sys.stderr)
    print(" - Are the files in 'data/'?")
    print(" - Are files .xlsx/.xls/.csv?")
    print(" - Do your files include at least some of: ", EXPECTED_COLUMNS)
    sys.exit(1)

# Concatenate
data = pd.concat(df_list, ignore_index=True, sort=False)
print(f"✅ Combined dataset shape: {data.shape}")
print("Columns in combined dataset:", list(data.columns))
print("Sample rows:\n", data.head())

# --------------------------
# Basic cleaning
# --------------------------
# Convert numeric columns to numeric where possible
numeric_cols = [
    "age", "blood_pressure", "diastolic", "blood_sugar", 
    "body_temp", "bmi", "heart_rate"
]
num_cols = [c for c in numeric_cols if c in data.columns]
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')

# Convert binary columns to 0/1
binary_cols = [
    "previous_complications", "preexisting_diabetes", 
    "gestational_diabetes", "mental_health"
]
binary_present = [c for c in binary_cols if c in data.columns]
if binary_present:
    data[binary_present] = data[binary_present].fillna(0).astype(int)

# Drop rows where target is missing
data = data.dropna(subset=["risk"])
# Optionally drop rows where all numeric features missing
data = data.dropna(subset=num_cols, how="all")

# Fill numeric NaNs with median
for c in num_cols:
    if c in data.columns and data[c].isna().any():
        median = data[c].median()
        data[c] = data[c].fillna(median)

print("After cleaning, dataset shape:", data.shape)

# --------------------------
# Encode target and features
# --------------------------
le = LabelEncoder()
data["risk"] = le.fit_transform(data["risk"].astype(str))

# Use all available feature columns for the model
feature_cols = num_cols + binary_present
X = data[feature_cols]
y = data["risk"]

# If X is empty or too small -> stop
if X.shape[0] < 10:
    print("❌ Not enough rows to train (need >= 10). Rows:", X.shape[0], file=sys.stderr)
    sys.exit(1)

# --------------------------
# Train / Eval
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n✅ Model Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Calculate feature importance
feature_importance = pd.DataFrame(
    {'feature': feature_cols, 'importance': model.feature_importances_}
).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Save things
joblib.dump(model, "pregnancy_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")  # Save feature columns for inference
data.to_csv("combined_pregnancy_data.csv", index=False)
print("\n🎯 Model and artifacts saved (pregnancy_model.pkl, scaler.pkl, label_encoder.pkl, feature_columns.pkl, combined_pregnancy_data.csv)")