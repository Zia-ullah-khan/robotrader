import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset that was just generated
dataset_path = '../datasets/AAPL_dataset.csv'
df = pd.read_csv(dataset_path)

# Remove Date column and empty columns
if 'Date' in df.columns:
    df = df.drop(columns=['Date'])
if '' in df.columns:
    df = df.drop(columns=[''])

# Remove target column to get feature columns
TARGET = 'realized_volatility'
if TARGET in df.columns:
    feature_cols = [col for col in df.columns if col != TARGET]
    feature_data = df[feature_cols]
else:
    print(f"Warning: {TARGET} not found in dataset columns: {df.columns.tolist()}")
    # Try to find volatility columns
    volatility_cols = [col for col in df.columns if 'volatility' in col.lower()]
    if volatility_cols:
        print(f"Found volatility columns: {volatility_cols}")
        TARGET = volatility_cols[0]  # Use the first volatility column
        feature_cols = [col for col in df.columns if col != TARGET]
        feature_data = df[feature_cols]
    else:
        print("No volatility columns found. Using all columns for scaler.")
        feature_data = df
        feature_cols = df.columns.tolist()

print(f"Feature columns: {feature_cols}")
print(f"Feature data shape: {feature_data.shape}")

# Create and fit the scaler
scaler = StandardScaler()
scaler.fit(feature_data)

# Save the scaler
scaler_path = './cnnLstmAttenScaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler recreated and saved to {scaler_path}")

# Test the scaler
test_transform = scaler.transform(feature_data[:5])
print(f"Test transform shape: {test_transform.shape}")
print("Scaler recreation successful!")
