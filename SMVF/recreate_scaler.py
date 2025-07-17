import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
dataset_path = '../datasets/AAPL_dataset.csv'
df = pd.read_csv(dataset_path)
if 'Date' in df.columns:
    df = df.drop(columns=['Date'])
if '' in df.columns:
    df = df.drop(columns=[''])
TARGET = 'realized_volatility'
if TARGET in df.columns:
    feature_cols = [col for col in df.columns if col != TARGET]
    feature_data = df[feature_cols]
else:
    print(f"Warning: {TARGET} not found in dataset columns: {df.columns.tolist()}")
    volatility_cols = [col for col in df.columns if 'volatility' in col.lower()]
    if volatility_cols:
        print(f"Found volatility columns: {volatility_cols}")
        TARGET = volatility_cols[0]
        feature_cols = [col for col in df.columns if col != TARGET]
        feature_data = df[feature_cols]
    else:
        print("No volatility columns found. Using all columns for scaler.")
        feature_data = df
        feature_cols = df.columns.tolist()
print(f"Feature columns: {feature_cols}")
print(f"Feature data shape: {feature_data.shape}")
scaler = StandardScaler()
scaler.fit(feature_data)
scaler_path = './cnnLstmAttenScaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"Scaler recreated and saved to {scaler_path}")
test_transform = scaler.transform(feature_data[:5])
print(f"Test transform shape: {test_transform.shape}")
print("Scaler recreation successful!")
