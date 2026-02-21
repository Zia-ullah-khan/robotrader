
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Add SMVF to path for dataset generation if needed
sys.path.append(os.path.join(os.getcwd(), 'SMVF'))
from dataset import generate_dataset

# Configuration
TICKER = "^GSPC"
DATASET_FILE = os.path.join("datasets", "GSPC_dataset.csv")
MODEL_PATH = os.path.join("SMVF", "cnn_lstm_attention_volatility.keras")
START_DATE = "2019-01-01"
END_DATE = "2024-12-31"
WINDOW_SIZE = 20
MC_DROPOUT_ITERATIONS = 50

# Custom Metric for Model Loading
@register_keras_serializable()
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def ensure_dataset():
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset {DATASET_FILE} not found. Generating...")
        generate_dataset([TICKER], "2010-01-01", datetime.now().strftime("%Y-%m-%d"))
    else:
        print(f"Dataset {DATASET_FILE} found.")

def prepare_data(df):
    # Standard cleaning
    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        df_numeric = df.drop(columns=['Date'])
    else:
        # Assuming index is date if no Date column, but dataset.py usually creates one
        dates = pd.to_datetime(df.index)
        df_numeric = df
        
    if '' in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=[''])

    # Fix column names to match training preprocessing
    df_numeric.columns = df_numeric.columns.str.replace(r'_5_2\.0_2\.0', '_5_2.0', regex=True)
    df_numeric.columns = df_numeric.columns.str.replace(r'STOCHh_14_3_3', 'STOCHk_14_3_3', regex=True)
    
    return dates, df_numeric

def create_sequences(data, window_size):
    # create rolling windows
    # data is expected to be numpy array of features
    num_samples = len(data) - window_size
    if num_samples <= 0:
        return np.array([])
    
    # We want X to be [0..19], [1..20], ...
    # numpy stride tricks could be faster but simple loop is robust for now
    X = []
    for i in range(num_samples):
        window = data[i : i + window_size]
        X.append(window)
    return np.array(X)

def main():
    ensure_dataset()
    
    print("Loading data...")
    df = pd.read_csv(DATASET_FILE)
    all_dates, df_numeric = prepare_data(df)
    
    # We need to scale the data. 
    # To correspond with 'compare_volatility_models', we fit on the full available dataset 
    # (or strictly, one should fit on train, but for visualization of full period we often scale globally in these simple scripts)
    scaler = StandardScaler()
    feature_cols = [c for c in df_numeric.columns if c != 'realized_volatility']
    target_col = 'realized_volatility'
    
    if target_col not in df_numeric.columns:
        print(f"Error: {target_col} not in dataset")
        return

    df_numeric[feature_cols] = scaler.fit_transform(df_numeric[feature_cols])
    
    # Feature array and Target array
    data_values = df_numeric[feature_cols].values
    target_values = df_numeric[target_col].values
    
    # Create windows for the WHOLE dataset first, then filter by date, 
    # to ensure we have the correct window for the first date of our interest period.
    print("Creating sequences...")
    X_all = create_sequences(data_values, WINDOW_SIZE)
    # y corresponds to the target at the END of the window? 
    # In 'dataset.py', realized_volatility is calculated using rolling window.
    # In 'trainCnnLstmAttn.py':
    # for i in range(len(data) - window_size):
    #     X.append(data... i:i+window_size)
    #     y.append(data... i+window_size)
    # So X[k] predicts y[k]. y[k] is at index `i + window_size`.
    # Corresponding date is also at `i + window_size`.
    
    # Adjust dates
    # sequences start at index 0 (using rows 0 to 19). Target is row 20.
    # So valid targets start from index WINDOW_SIZE.
    valid_dates = all_dates[WINDOW_SIZE:]
    valid_targets = target_values[WINDOW_SIZE:]
    
    # Ensure alignment
    # X_all has length len(df) - WINDOW_SIZE.
    # valid_dates has length len(df) - WINDOW_SIZE.
    
    if len(X_all) != len(valid_dates):
        print(f"Shape mismatch: X={len(X_all)}, Dates={len(valid_dates)}")
        return

    # Filter for the requested period: 2019-01-01 to 2024-12-31
    mask = (valid_dates >= START_DATE) & (valid_dates <= END_DATE)
    
    X_test = X_all[mask]
    y_test = valid_targets[mask]
    dates_test = valid_dates[mask]
    
    print(f"Analysis Period: {START_DATE} to {END_DATE}")
    print(f"Number of samples: {len(X_test)}")
    
    if len(X_test) == 0:
        print("No data found for the specified period.")
        return

    print("Loading Model...")
    try:
        model = load_model(MODEL_PATH, custom_objects={'rmse': rmse})
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Running Monte Carlo Dropout Inference ({MC_DROPOUT_ITERATIONS} iterations)...")
    
    # Convert to tensor for efficiency in calling model(..., training=True)
    X_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    
    mc_predictions = []
    
    # Run loop
    # We can do this in batches if memory is an issue, but 1500 samples is small enough for one batch.
    for i in range(MC_DROPOUT_ITERATIONS):
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{MC_DROPOUT_ITERATIONS}")
        
        # training=True enables Dropout during inference
        preds = model(X_tensor, training=False) 
        # WAIT! The prompt says "confidence intervals generated by model uncertainty". 
        # Usually this implies MC Dropout (training=True).
        # However, the user says "Values from the hybrid model are well aligned... confidence intervals... get wider".
        # If I use training=False, I get deterministic output (std=0).
        # So I MUST use training=True.
        preds = model(X_tensor, training=True)
        mc_predictions.append(preds.numpy().flatten())
        
    mc_predictions = np.array(mc_predictions) # shape (50, n_samples)
    
    # Calculate Statistics
    mean_preds = np.mean(mc_predictions, axis=0)
    std_preds = np.std(mc_predictions, axis=0)
    
    lower_bound = mean_preds - 2 * std_preds
    upper_bound = mean_preds + 2 * std_preds
    
    # Plotting
    print("Generating Plot...")
    plt.figure(figsize=(14, 7))
    
    # Plot slightly thinner lines for aesthetic
    plt.plot(dates_test, y_test, label='Realized Volatility (Ground Truth)', color='black', linewidth=1, alpha=0.8)
    plt.plot(dates_test, mean_preds, label='Forecast (Hybrid Model)', color='#1f77b4', linewidth=1.5)
    
    # Shaded Confidence Interval
    plt.fill_between(dates_test, lower_bound, upper_bound, color='#1f77b4', alpha=0.3, label='Model Uncertainty (95% CI)')
    
    plt.title('Figure 6: Forecast Volatility Bands with Confidence Shading – S&P 500 (2019-2024)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Highlight specific volatile periods if desired (e.g. Covid 2020)
    # plt.axvspan(pd.to_datetime('2020-03-01'), pd.to_datetime('2020-04-30'), color='red', alpha=0.1, label='COVID-19 Crash')

    output_path = "forecast_volatility_bands_2019_2024.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also save the data mainly for user inspection if needed
    results_df = pd.DataFrame({
        'Date': dates_test,
        'Actual': y_test,
        'Forecast': mean_preds,
        'Uncertainty_Std': std_preds
    })
    results_df.to_csv("forecast_data_export.csv", index=False)
    print("Underlying data saved to forecast_data_export.csv")

if __name__ == "__main__":
    main()
