import pandas as pd
import numpy as np
import os

print("Loading dataset...")
# Read the dataset
df = pd.read_csv("Dataset - Updated.csv")

# Display original shape
print(f"Original dataset shape: {df.shape}")

# ------------ Data Cleaning ------------

# 1. Check for and fix data types
print("\nChecking data types...")
print(df.dtypes)

# 2. Handle missing values
print("\nMissing values before cleaning:")
print(df.isna().sum())

# Replace empty strings with NaN
df = df.replace('', np.nan)

# Fix incorrect age value (325 is likely 32)
if 'Age' in df.columns:
    df.loc[df['Age'] > 100, 'Age'] = 32

# 3. Fill missing numeric values with median
numeric_cols = ['Age', 'Systolic BP', 'Diastolic', 'BS', 'BMI', 'Heart Rate']
for col in numeric_cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {col} NAs with median: {median_val}")

# 4. Standardize risk level (make sure it's always filled)
if 'Risk Level' in df.columns:
    # Fill any missing risk levels based on similar rows
    # This is a simple approach - could be improved
    df['Risk Level'] = df['Risk Level'].fillna('Low')
    
    # Standardize risk level values
    df['Risk Level'] = df['Risk Level'].str.strip()
    print(f"Risk level values: {df['Risk Level'].unique()}")

# 5. Remove outliers or cap extreme values
if 'BS' in df.columns:  # Blood Sugar
    # Cap extreme blood sugar values at reasonable limits
    q1 = df['BS'].quantile(0.25)
    q3 = df['BS'].quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr
    
    df.loc[df['BS'] > upper_limit, 'BS'] = upper_limit
    df.loc[df['BS'] < lower_limit, 'BS'] = lower_limit
    print(f"Capped blood sugar values between {lower_limit:.2f} and {upper_limit:.2f}")

# 6. Check for remaining missing values after cleaning
print("\nMissing values after initial cleaning:")
print(df.isna().sum())

# 7. Drop rows that still have too many missing values
# If a row is missing more than 40% of values, drop it
threshold = int(0.4 * len(df.columns))
df = df.dropna(thresh=len(df.columns) - threshold)
print(f"After dropping rows with too many missing values: {df.shape}")

# 8. Final cleanup for any remaining missing values
df = df.fillna(df.median(numeric_only=True))
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Show final stats
print("\nFinal dataset statistics:")
print(df.describe())

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Save the cleaned dataset to the data folder
cleaned_file = "data/cleaned_pregnancy_dataset.csv"
df.to_csv(cleaned_file, index=False)
print(f"\nCleaned data saved to {cleaned_file}")
print(f"Final dataset shape: {df.shape}")

# Show sample rows
print("\nSample from cleaned dataset:")
print(df.head())