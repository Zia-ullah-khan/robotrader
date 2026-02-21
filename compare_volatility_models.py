
import os
import sys
import pandas as pd
import numpy as np
import arch
import joblib
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Add SMVF to path
sys.path.append(os.path.join(os.getcwd(), 'SMVF'))
from dataset import generate_dataset

# Configuration
TICKER = "^GSPC"
DATASET_DIR = "datasets"
DATASET_FILE = "GSPC_dataset.csv"
DATASET_PATH = os.path.join(DATASET_DIR, DATASET_FILE)
MODEL_PATH = os.path.join("SMVF", "cnn_lstm_attention_volatility.keras")
WINDOW_SIZE = 20

# Metric for custom model loading
@register_keras_serializable()
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def ensure_dataset():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset {DATASET_PATH} not found. Generating...")
        # Start date enough to capture 2018-2024 plus buffer
        generate_dataset([TICKER], "2010-01-01", "2024-12-31")
    else:
        print(f"Dataset {DATASET_PATH} found.")

def prepare_data(df):
    # Preprocessing matching the training script
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        df_numeric = df.drop(columns=['Date'])
    else:
        dates = df.index
        df_numeric = df
        
    if '' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=[''])

    # Fix column names if needed (regex from predict.py)
    df_numeric.columns = df_numeric.columns.str.replace(r'_5_2\.0_2\.0', '_5_2.0', regex=True)
    df_numeric.columns = df_numeric.columns.str.replace(r'STOCHh_14_3_3', 'STOCHk_14_3_3', regex=True)
    
    return dates, df_numeric

def run_comparison():
    ensure_dataset()
    
    # Load Data
    df_full = pd.read_csv(DATASET_PATH)
    dates, df_numeric = prepare_data(df_full)
    
    # Identify Ultra Rare Events (Highest Realized Volatility)
    # We focus on the period user mentioned: 2018-01-01 to 2024-12-31
    # Filter by date first
    mask = (dates >= "2018-01-01") & (dates <= "2024-12-31")
    analysis_indices = df_numeric.loc[mask].index
    
    # Find top 5 volatility events in this range
    top_vol_indices = df_numeric.loc[analysis_indices, 'realized_volatility'].nlargest(5).index
    
    print("\n--- Identifying Ultra Rare (High Volatility) Events ---")
    for idx in top_vol_indices:
        print(f"Date: {dates[idx].date()}, Volatility: {df_numeric.loc[idx, 'realized_volatility']:.4f}")

    # Load Custom Model
    print("\nLoading Custom CNN-LSTMagent Model...")
    try:
        model = load_model(MODEL_PATH, custom_objects={'rmse': rmse})
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Scale Data (Fit on entire dataset to mimic training environment assumption)
    scaler = StandardScaler()
    feature_cols = [c for c in df_numeric.columns if c != 'realized_volatility']
    
    # We need to scale only feature columns
    # Note: Training logic scales ALL columns except target? 
    # trainCnnLstmAttn.py lines 44-45:
    # feature_cols = [col for col in df.columns if col != TARGET]
    # df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # We do the same here
    df_scaled_features = df_numeric.copy()
    df_scaled_features[feature_cols] = scaler.fit_transform(df_numeric[feature_cols])

    comparison_results = []

    print("\n--- Running Predictions ---")
    
    for idx in top_vol_indices:
        date_str = dates[idx].date()
        
        # 1. Custom Model Prediction
        # Needs window [idx-WINDOW_SIZE : idx] -> indices idx-20 to idx-1 (20 items)
        # Using scaled features
        if idx < WINDOW_SIZE:
            print(f"Skipping {date_str}: Insufficient history.")
            continue
            
        # Extract window
        input_window = df_scaled_features[feature_cols].iloc[idx-WINDOW_SIZE:idx].values
        input_tensor = np.expand_dims(input_window, axis=0)
        
        custom_pred = model.predict(input_tensor, verbose=0).ravel()[0]
        
        # 2. GARCH Prediction
        # Uses returns up to idx-1 to predict idx
        # We need the 'log_returns' column.
        if 'log_returns' not in df_numeric.columns:
            print("Error: 'log_returns' column missing.")
            continue
            
        returns_history = df_numeric['log_returns'].iloc[:idx].dropna() # Returns up to t-1?
        # Note: df['log_returns'][k] is return at time k.
        # To predict volatility at time `idx` (which uses returns up to `idx`), 
        # we realistically only have info up to `idx-1` to make the prediction "ex-ante".
        # If we use return at `idx`, we assume we know the return of the day.
        # But `realized_volatility` at `idx` is computed using `log_returns` at `idx`.
        # So "Realized Volatility" is an ex-post measure.
        # The prediction should be made at `idx-1`.
        # So GARCH should be fitted on returns up to `idx-1`.
        
        returns_history = df_numeric['log_returns'].iloc[:idx] # 0 to idx-1
        returns_history = returns_history.dropna()
        
        if len(returns_history) < 252: # Require at least a year of history for stable GARCH
             print(f"Skipping GARCH for {date_str}: Insufficient history ({len(returns_history)}).")
             garch_pred = np.nan
        else:
             # Scale returns for GARCH stability (standard practice involves *100)
             returns_rescaled = returns_history * 100
             
             try:
                 # GARCH(1,1) with constant mean
                 garch_model = arch.arch_model(returns_rescaled, p=1, q=1, vol='Garch', dist='Normal')
                 res = garch_model.fit(disp='off', last_obs=None)
                 
                 # Forecast next step
                 forecast = res.forecast(horizon=1)
                 # Get projected variance for the next step (which corresponds to 'idx')
                 # forecast.variance is indexed by the last date in returns_rescaled
                 next_val_var = forecast.variance.iloc[-1, 0]
                 
                 # Transform back: sqrt(var)/100 * sqrt(252)
                 garch_pred_daily = np.sqrt(next_val_var) / 100
                 garch_pred = garch_pred_daily * np.sqrt(252)
                 
             except Exception as e:
                 print(f"GARCH failed for {date_str}: {e}")
                 garch_pred = np.nan

        comparison_results.append({
            "Date": date_str,
            "Actual_Vol": df_numeric.loc[idx, 'realized_volatility'],
            "Classic_GARCH": garch_pred,
            "My_Model": custom_pred
        })

    # Display Results
    results_df = pd.DataFrame(comparison_results)
    results_df = results_df.sort_values("Date")
    
    print("\n" + "="*80)
    print("COMPARISON: Classic GARCH(1,1) vs. Custom Attention Model on Extreme Events")
    print("="*80)
    # Reorder for nice printing
    results_df = results_df[["Date", "Actual_Vol", "Classic_GARCH", "My_Model"]]
    # Add difference columns
    results_df['GARCH_Error'] = abs(results_df['Actual_Vol'] - results_df['Classic_GARCH'])
    results_df['MyModel_Error'] = abs(results_df['Actual_Vol'] - results_df['My_Model'])
    
    print(results_df.to_string(index=False, float_format="%.4f"))
    
    print("\nSummary:")
    avg_garch_err = results_df['GARCH_Error'].mean()
    avg_my_err = results_df['MyModel_Error'].mean()
    print(f"Average Error (MAE) - GARCH: {avg_garch_err:.4f}")
    print(f"Average Error (MAE) - My Model: {avg_my_err:.4f}")
    
    if avg_my_err < avg_garch_err:
        print("Conclusion: My Model outperformed GARCH on these extreme events.")
    else:
        print("Conclusion: GARCH outperformed (or results are mixed).")

if __name__ == "__main__":
    run_comparison()
