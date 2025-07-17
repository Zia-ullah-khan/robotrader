import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Multiply, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K
from keras.saving import register_keras_serializable
import joblib

DATA_PATH = 'datasets/GSPC_dataset.csv'
WINDOW_SIZE = 20
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.3
TARGET = 'realized_volatility'

df = pd.read_csv(DATA_PATH)
if 'Date' in df.columns:
    df = df.drop(columns=['Date'])
if '' in df.columns:
    df = df.drop(columns=[''])

scaler = StandardScaler()
feature_cols = [col for col in df.columns if col != TARGET]
df[feature_cols] = scaler.fit_transform(df[feature_cols])
joblib.dump(scaler, './cnnLstmAtten/cnnLstmAttenScaler.pkl')
print("Scaler saved as cnnLstmAttenScaler.pkl")

def create_rolling_windows(data, window_size, target_col):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i+window_size].values)
        y.append(data.iloc[i+window_size][target_col])
    return np.array(X), np.array(y)

X, y = create_rolling_windows(df, WINDOW_SIZE, TARGET)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

@register_keras_serializable()
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def build_model(input_shape):
    inp = Input(shape=input_shape)  # (window_size, num_features)
    
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    x = LSTM(64, return_sequences=True)(x)  # (batch, window_size, 64)
    
    score = Dense(1, activation='tanh')(x)  # (batch, window_size, 1)
    score = Flatten()(score)                # (batch, window_size)
    attn_weights = Dense(WINDOW_SIZE, activation='softmax')(score)  # (batch, window_size)
    attn_weights = attn_weights[..., None]  # (batch, window_size, 1)
    
    attn_applied = Multiply()([x, attn_weights])  # (batch, window_size, 64)
    
    x = Flatten()(attn_applied)
    x = Dense(64, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inp, outputs=out)
    return model

model = build_model(X_train.shape[1:])
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mae', rmse])

callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

model.save('./cnnLstmAtten/cnn_lstm_attention_volatility.keras')
print('Training complete. Model saved as cnn_lstm_attention_volatility.keras')

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.show()
