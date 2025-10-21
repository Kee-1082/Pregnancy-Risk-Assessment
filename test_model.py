import joblib
import pandas as pd
import numpy as np

# Load model artifacts
model = joblib.load("pregnancy_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_cols = joblib.load("feature_columns.pkl")

print("Model loaded successfully!")
print(f"Feature columns: {feature_cols}")
print(f"Risk classes: {label_encoder.classes_}")

# Test with diverse inputs
test_cases = [
    {
        "name": "Low Risk Case",
        "age": 25, "blood_pressure": 110, "diastolic": 70, "blood_sugar": 7.0,
        "body_temp": 98.6, "bmi": 22.0, "heart_rate": 72, "hemoglobin": 12.5
    },
    {
        "name": "High Risk Case - High BP",
        "age": 38, "blood_pressure": 150, "diastolic": 95, "blood_sugar": 9.5,
        "body_temp": 99.0, "bmi": 32.0, "heart_rate": 88, "hemoglobin": 10.5,
        "chronic_hypertension": 1, "advanced_maternal_age": 1
    },
    {
        "name": "Moderate Risk Case",
        "age": 30, "blood_pressure": 125, "diastolic": 80, "blood_sugar": 8.0,
        "body_temp": 98.8, "bmi": 26.0, "heart_rate": 78, "hemoglobin": 11.8
    }
]

print("\n" + "="*60)
print("TESTING MODEL WITH DIVERSE INPUTS")
print("="*60)

for test_case in test_cases:
    name = test_case.pop("name")
    
    # Create dataframe
    input_df = pd.DataFrame([test_case])
    
    # Add missing features as 0
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Reorder columns
    input_df = input_df[feature_cols]
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    risk_level = label_encoder.inverse_transform([prediction])[0]
    
    print(f"\n{name}:")
    print(f"  Input: {test_case}")
    print(f"  Prediction: {risk_level}")
    print(f"  Probabilities: {dict(zip(label_encoder.classes_, probabilities))}")
    print(f"  Confidence: {max(probabilities)*100:.1f}%")