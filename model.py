# model.py
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ========================================
# IMPORTING ALL 5 ML ALGORITHMS
# ========================================
from sklearn.ensemble import RandomForestClassifier      # Random Forest Algorithm
from sklearn.svm import SVC                              # Support Vector Machine (RBF)
from sklearn.tree import DecisionTreeClassifier         # Decision Tree Algorithm
from sklearn.naive_bayes import GaussianNB              # Gaussian Naive Bayes Algorithm
import xgboost as xgb                                    # XGBoost Algorithm

import joblib
import sys

# --------------------------
# Config
# --------------------------
DATA_DIR = Path("data")

# SIMPLIFIED EXPECTED COLUMNS (removed social factors, excessive substance use, and complex labs)
EXPECTED_COLUMNS = [
    # Basic Demographics & Vitals
    "age", "blood_pressure", "diastolic", "blood_sugar", "body_temp", 
    "bmi", "heart_rate",
    
    # Key Medical Conditions
    "asthma", "preexisting_diabetes", "gestational_diabetes", 
    "advanced_maternal_age", "chronic_hypertension", 
    "pregnancy_induced_hypertension", "genetic_disorder", "mental_health",
    
    # Important Obstetric History
    "previous_complications", "previous_preterm_birth", 
    "current_placental_problems", "medications_affecting_fetus",
    "multiple_gestation",
    
    # Critical Substance Use (simplified)
    "alcohol_use", "tobacco_use", "narcotics_use", "sedatives_use",
    
    # Nutrition & Additional
    "pregnancy_weight_issue", "obstetric_diet_condition", 
    "anemia", "cervical_length", "hemoglobin",
    
    # Target
    "risk"
]

# Simplified Column Mapping
COLUMN_MAP = {
    # Basic vitals
    "age": ["age", "age_years", "years"],
    "blood_pressure": ["blood_pressure", "bp", "bloodpressure", "systolic_bp", "systolic", "systolic blood pressure"],
    "diastolic": ["diastolic", "diastolic_bp", "dbp", "diastolic blood pressure"],
    "blood_sugar": ["blood_sugar", "bloodsugar", "glucose", "sugar", "bs", "blood sugar"],
    "body_temp": ["body_temp", "temperature", "body_temperature", "temp", "body temperature"],
    "bmi": ["bmi", "body_mass_index", "body mass index"],
    "heart_rate": ["heart_rate", "hr", "pulse", "heartrate", "heart rate"],
    
    # Medical conditions
    "asthma": ["asthma", "respiratory_condition"],
    "preexisting_diabetes": ["preexisting_diabetes", "diabetes", "pre_diabetes", "preexisting diabetes"],
    "gestational_diabetes": ["gestational_diabetes", "gest_diabetes", "gestational diabetes"],
    "advanced_maternal_age": ["advanced_maternal_age", "maternal_age_35_plus", "age_over_35", "advanced maternal age"],
    "chronic_hypertension": ["chronic_hypertension", "htn", "chronic_htn", "preexisting_hypertension", "chronic hypertension"],
    "pregnancy_induced_hypertension": ["pregnancy_induced_hypertension", "pih", "gestational_hypertension", "pregnancy induced hypertension"],
    "genetic_disorder": ["genetic_disorder", "genetic_condition", "hereditary_disorder", "genetic disorder"],
    "mental_health": ["mental_health", "mental", "mental health"],
    
    # Obstetric history
    "previous_complications": ["previous_complications", "complications", "prev_complications", "previous complications"],
    "previous_preterm_birth": ["previous_preterm_birth", "preterm_history", "early_delivery_history", "previous preterm birth"],
    "current_placental_problems": ["current_placental_problems", "placenta_previa", "placental_issues", "current placental problems"],
    "medications_affecting_fetus": ["medications_affecting_fetus", "teratogenic_meds", "harmful_medications", "medications affecting fetus"],
    "multiple_gestation": ["multiple_gestation", "twins", "triplets", "multiple_babies", "multiple gestation"],
    
    # Substance use (simplified)
    "alcohol_use": ["alcohol_use", "alcohol_consumption", "drinking", "alcohol", "alcohol use"],
    "tobacco_use": ["tobacco_use", "smoking", "cigarettes", "tobacco", "tobacco use"],
    "narcotics_use": ["narcotics_use", "opioid_use", "heroin_use", "narcotics", "narcotics use"],
    "sedatives_use": ["sedatives_use", "tranquilizer_use", "benzodiazepines", "sedatives", "sedatives use"],
    
    # Nutrition & additional
    "pregnancy_weight_issue": ["pregnancy_weight_issue", "weight_gain_issue", "inadequate_weight_gain", "pregnancy weight issue"],
    "obstetric_diet_condition": ["obstetric_diet_condition", "diet_modification_needed", "special_diet", "obstetric diet condition"],
    "anemia": ["anemia", "anaemia", "low_hemoglobin"],
    "cervical_length": ["cervical_length", "cx_length", "cervical length"],
    "hemoglobin": ["hemoglobin", "haemoglobin", "hb"],
    
    # Target
    "risk": ["risk", "label", "outcome", "target", "risk_level", "risk level"]
}

def canonicalize_columns(df):
    """Normalize column names and map to canonical names"""
    df = df.copy()
    # Lowercase, strip spaces, replace spaces with underscores
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
            df = pd.read_excel(f)
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

    # Check for minimum required columns
    required_columns = ["age", "blood_pressure", "blood_sugar", "risk"]
    present = [c for c in required_columns if c in df.columns]
    if len(present) < len(required_columns):
        print(f"   ⚠️ File {f.name} doesn't have required columns (found: {present}, need: {required_columns}), skipping.")
        continue

    # Keep only expected columns that are present
    keep = [c for c in EXPECTED_COLUMNS if c in df.columns]
    df = df[keep]
    df_list.append(df)

# If nothing read, stop with helpful message
if not df_list:
    print("❌ No valid dataframes to concatenate. Check the 'data' folder and file columns.", file=sys.stderr)
    print(" - Are the files in 'data/'?")
    print(" - Are files .xlsx/.xls/.csv?")
    print(" - Do your files include at least: age, blood_pressure, blood_sugar, risk")
    sys.exit(1)

# Concatenate all dataframes
data = pd.concat(df_list, ignore_index=True, sort=False)
print(f"✅ Combined dataset shape: {data.shape}")
print("Columns in combined dataset:", list(data.columns))
print("Sample rows:\n", data.head())

# --------------------------
# Basic cleaning
# --------------------------
# Define numeric and binary columns
numeric_cols = [
    "age", "blood_pressure", "diastolic", "blood_sugar", 
    "body_temp", "bmi", "heart_rate", "cervical_length", "hemoglobin"
]

binary_cols = [
    # Medical conditions
    "asthma", "preexisting_diabetes", "gestational_diabetes", 
    "advanced_maternal_age", "chronic_hypertension", 
    "pregnancy_induced_hypertension", "genetic_disorder", "mental_health",
    
    # Obstetric history
    "previous_complications", "previous_preterm_birth", 
    "current_placental_problems", "medications_affecting_fetus",
    "multiple_gestation",
    
    # Substance use
    "alcohol_use", "tobacco_use", "narcotics_use", "sedatives_use",
    
    # Nutrition & additional
    "pregnancy_weight_issue", "obstetric_diet_condition", "anemia"
]

# Convert numeric columns
num_cols = [c for c in numeric_cols if c in data.columns]
data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')

# Convert binary columns with safer mapping
binary_present = [c for c in binary_cols if c in data.columns]
if binary_present:
    for col in binary_present:
        data[col] = data[col].map(
            {
                # True values
                True: 1, 'True': 1, 'true': 1, 'TRUE': 1,
                't': 1, 'T': 1, 'yes': 1, 'Yes': 1, 'YES': 1,
                'y': 1, 'Y': 1, '1': 1, 1: 1,
                
                # False values
                False: 0, 'False': 0, 'false': 0, 'FALSE': 0,
                'f': 0, 'F': 0, 'no': 0, 'No': 0, 'NO': 0,
                'n': 0, 'N': 0, '0': 0, 0: 0
            }, na_action='ignore'
        ).fillna(0).astype(int)

# Drop rows where target is missing
data = data.dropna(subset=["risk"])

# Drop rows where all numeric features are missing
data = data.dropna(subset=num_cols, how="all")

# Fill numeric NaNs with median
for c in num_cols:
    if c in data.columns and data[c].isna().any():
        median = data[c].median()
        data[c] = data[c].fillna(median)

print("After cleaning, dataset shape:", data.shape)

# --------------------------
# Normalize risk labels
# --------------------------
# Standardize risk level values
high_risk_values = ['high', 'high risk', 'h', '1', 'risk', 'risky', 'yes']
low_risk_values = ['low', 'low risk', 'l', '0', 'normal', 'safe', 'no']

# Convert risk to lowercase for easier mapping
data['risk'] = data['risk'].astype(str).str.lower().str.strip()

# Map to standard values
data['risk'] = data['risk'].apply(
    lambda x: 'High' if x in high_risk_values else 
             ('Low' if x in low_risk_values else x)
)

print(f"Unique risk values after normalization: {data['risk'].unique()}")

# --------------------------
# Encode target and features
# --------------------------
le = LabelEncoder()
data["risk"] = le.fit_transform(data["risk"])

# Use all available feature columns for the model
feature_cols = num_cols + binary_present
X = data[feature_cols]
y = data["risk"]

# Check if we have enough data
if X.shape[0] < 10:
    print("❌ Not enough rows to train (need >= 10). Rows:", X.shape[0], file=sys.stderr)
    sys.exit(1)

print(f"\n📊 Feature Summary:")
print(f"Total features: {len(feature_cols)}")
print(f"Numeric features: {len(num_cols)}")
print(f"Binary features: {len(binary_present)}")
print(f"Features used: {feature_cols}")

# --------------------------
# Train / Test Split
# --------------------------
print(f"\n📊 Class Distribution Analysis:")
print(data['risk'].value_counts())
print(f"Percentage distribution:\n{data['risk'].value_counts(normalize=True) * 100}")

# Check for severe class imbalance
class_counts = data['risk'].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\n⚠️ Class imbalance ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 3:
    print("⚠️ WARNING: Severe class imbalance detected!")
    print("Applying stratified split and class weights...")

# Use stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n📈 Dataset Split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Training class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Testing class distribution:\n{pd.Series(y_test).value_counts()}")

# ========================================
# INITIALIZE ALL 5 ML ALGORITHMS WITH BALANCED PARAMETERS
# ========================================
print("\n🤖 Initializing Machine Learning Algorithms...")

# Calculate class weights for imbalanced data
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# 1. RANDOM FOREST ALGORITHM - with balanced weights
rf_model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42,
    class_weight='balanced',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

# 2. SUPPORT VECTOR MACHINE (RBF KERNEL) ALGORITHM - with balanced weights
svm_model = SVC(
    kernel='rbf', 
    random_state=42, 
    probability=True,
    class_weight='balanced',
    C=1.0,
    gamma='scale'
)

# 3. DECISION TREE ALGORITHM - with balanced weights
dt_model = DecisionTreeClassifier(
    random_state=42,
    class_weight='balanced',
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2
)

# 4. XGBOOST ALGORITHM - with scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)

# 5. GAUSSIAN NAIVE BAYES ALGORITHM
nb_model = GaussianNB()

# Store all models in a dictionary
models = {
    'Random Forest': rf_model,
    'Support Vector Machine (RBF)': svm_model, 
    'Decision Tree': dt_model,
    'XGBoost': xgb_model,
    'Gaussian Naive Bayes': nb_model
}

# ========================================
# TRAIN AND EVALUATE ALL 5 ALGORITHMS
# ========================================
print("\n🏋️ Training and Evaluating All Models...")
print("="*60)

model_results = {}
best_model = None
best_accuracy = 0
best_model_name = ""

for name, model in models.items():
    print(f"\n📊 Training {name}...")
    
    # ALGORITHM TRAINING
    if name in ['Support Vector Machine (RBF)', 'Gaussian Naive Bayes']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    # ALGORITHM EVALUATION
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate per-class accuracy
    from sklearn.metrics import balanced_accuracy_score, f1_score
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store results
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'predictions': y_pred,
        'prediction_probabilities': y_pred_proba,
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    # Track best performing model using balanced accuracy
    if balanced_acc > best_accuracy:
        best_accuracy = balanced_acc
        best_model = model
        best_model_name = name
    
    print(f"✅ {name} Accuracy: {accuracy:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc:.4f}")
    print(f"   F1 Score: {f1:.4f}")

# ========================================
# DISPLAY DETAILED RESULTS
# ========================================
print("\n" + "="*60)
print("📈 DETAILED RESULTS FOR ALL 5 ALGORITHMS")
print("="*60)

for name, results in model_results.items():
    print(f"\n🔍 {name.upper()} RESULTS:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Classification Report:\n{results['classification_report']}")
    print(f"Confusion Matrix:\n{results['confusion_matrix']}")
    
    # Show prediction distribution
    pred_dist = pd.Series(results['predictions']).value_counts()
    print(f"Prediction distribution: {pred_dist.to_dict()}")
    print("-" * 40)

# ========================================
# BEST PERFORMING ALGORITHM
# ========================================
print(f"\n🏆 BEST PERFORMING ALGORITHM: {best_model_name}")
print(f"🎯 Best Balanced Accuracy: {best_accuracy:.4f}")

# ========================================
# PREDICTION DIVERSITY CHECK
# ========================================
print("\n🔍 Prediction Diversity Analysis:")
for name, results in model_results.items():
    unique_preds = len(np.unique(results['predictions']))
    high_risk_pct = (results['predictions'] == 1).sum() / len(results['predictions']) * 100
    low_risk_pct = (results['predictions'] == 0).sum() / len(results['predictions']) * 100
    
    print(f"{name}:")
    print(f"  High Risk: {high_risk_pct:.1f}%")
    print(f"  Low Risk: {low_risk_pct:.1f}%")
    
    if high_risk_pct > 90 or low_risk_pct > 90:
        print(f"  ⚠️ WARNING: Model predicts {high_risk_pct if high_risk_pct > 90 else low_risk_pct:.1f}% as one class!")

# ========================================
# SAVE ALL MODELS AND ARTIFACTS
# ========================================
print(f"\n💾 Saving all models and artifacts...")

# Save the best performing model as the main model
joblib.dump(best_model, "pregnancy_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(feature_cols, "feature_columns.pkl")

# Save all models separately
for name, results in model_results.items():
    clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    joblib.dump(results['model'], f"model_{clean_name}.pkl")

# Save model comparison results with all metrics
comparison_df = pd.DataFrame([
    {
        'Algorithm': name, 
        'Accuracy': results['accuracy'],
        'Balanced_Accuracy': results['balanced_accuracy'],
        'F1_Score': results['f1_score']
    } 
    for name, results in model_results.items()
]).sort_values('Balanced_Accuracy', ascending=False)

comparison_df.to_csv("model_comparison_results.csv", index=False)
data.to_csv("combined_pregnancy_data.csv", index=False)

print(f"\n🎯 All models and artifacts saved!")
print(f"📁 Files saved:")
print(f"   - pregnancy_model.pkl (best model: {best_model_name})")
print(f"   - scaler.pkl")
print(f"   - label_encoder.pkl") 
print(f"   - feature_columns.pkl")
print(f"   - model_comparison_results.csv")
print(f"   - combined_pregnancy_data.csv")
print(f"   - Individual model files: model_*.pkl")

print(f"\n📊 Algorithm Comparison Summary:")
print(comparison_df.to_string(index=False))

# ========================================
# DATA QUALITY WARNINGS
# ========================================
print("\n" + "="*60)
print("⚠️ DATA QUALITY ASSESSMENT")
print("="*60)

if len(data) < 100:
    print("❌ CRITICAL: Dataset too small (< 100 samples)")
    print("   Recommendation: Collect more data")

if imbalance_ratio > 5:
    print(f"❌ CRITICAL: Severe class imbalance ({imbalance_ratio:.1f}:1)")
    print("   Recommendation: Balance your dataset or use SMOTE")

if accuracy_std < 0.02:
    print("⚠️ WARNING: All models perform similarly")
    print("   This suggests the problem may be too simple")

# Check if any model predicts > 90% of one class
for name, results in model_results.items():
    high_risk_pct = (results['predictions'] == 1).sum() / len(results['predictions']) * 100
    if high_risk_pct > 90 or high_risk_pct < 10:
        print(f"❌ WARNING: {name} predicts mostly one class ({high_risk_pct:.1f}% High Risk)")
        print("   Recommendation: Check data quality and feature engineering")

print("\n✅ Model training complete!")
