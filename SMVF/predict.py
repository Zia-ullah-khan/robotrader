import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import StandardScaler
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K # type: ignore
import joblib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@register_keras_serializable()
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

MODEL_PATH = os.path.join(BASE_DIR, 'cnn_lstm_attention_volatility.keras')
SCALER_PATH = os.path.join(BASE_DIR, 'cnnLstmAttenScaler.pkl')
WINDOW_SIZE = 20
TARGET = 'realized_volatility'
scaler = None
model = None

def predict_next_hour_volatility(ticker, dataset_csv):
    global model
    if model is None:
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    df = pd.read_csv(dataset_csv)
    df.columns = df.columns.str.replace(r'_5_2\.0_2\.0', '_5_2.0', regex=True)
    df.columns = df.columns.str.replace(r'STOCHh_14_3_3', 'STOCHk_14_3_3', regex=True)
    if 'Date' not in df.columns:
        raise ValueError("'Date' column not found in dataset")
    df = df.sort_values('Date')
    df = df.drop(columns=['Date'])
    if '' in df.columns:
        df = df.drop(columns=[''])
    if TARGET not in df.columns:
        raise ValueError(f"'{TARGET}' column not found in {dataset_csv}")

    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col != TARGET]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    data = df
    if len(data) < WINDOW_SIZE:
        raise ValueError("Not enough data for prediction")
    X_input = data[feature_cols].iloc[-WINDOW_SIZE:].values
    X_input = np.expand_dims(X_input, axis=0)
    predicted_volatility = model.predict(X_input).ravel()[0]
    return predicted_volatility