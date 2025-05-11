from models.DataHandler import get_percent_move_model_data
from sklearn.preprocessing import RobustScaler
import pandas as pd
import tensorflow as tf
import joblib
import numpy as np
import os

# --- Config ---
start_date = "2017-01-01"
up_threshold = 1.005
down_threshold = 0.995
model_type = "TfStraightUpDownModel"
version_base = "v0.1"
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# --- Data Loading ---
data = get_percent_move_model_data(
    start_date=start_date,
    ticker="SPY",
    up_threshold=up_threshold,
    down_threshold=down_threshold,
    skip_move_status=True
)

# --- Feature Selection ---
features = [
    'premarket_pct_change',
    # 'last_pm_9ma_diff',
    # 'last_pm_20ma_diff',
    # 'daily_9ma_diff',
    # 'daily_20ma_diff',
    'daily_ma9_slope',
    'daily_ma20_slope',
    '5min_premarket_9ma_slope',
    '5min_premarket_20ma_slope',
    'bb_position',
    'realized_volatility',
    'vix_open', 'vix_close',
    'us10y_open', 'us10y_close',
    'pm_MA9', 'pm_MA20',
]
# Target: (close-open)/open * 100
if 'close' not in data.columns or 'open' not in data.columns:
    raise ValueError("Data must contain 'close' and 'open' columns for target calculation.")
data['open_close_pct'] = ((data['close'] - data['open']) / data['open']) * 100
# --- Target: Up/Down Classification ---
# Predict 1 if open->close move is positive (call/up), 0 if negative (put/down)
data['call_put'] = (data['open_close_pct'] > 0).astype(int)
target = 'call_put'

# --- Data Cleaning ---
data = data.dropna(subset=features + [target])

# --- Train/Test Split ---
split_date = pd.to_datetime("2022-01-01").date()
train_data = data[data.index < split_date]
test_data = data[data.index >= split_date]

# --- Validation Split ---
val_split_idx = int(len(train_data) * 0.9)
X_train = train_data[features].iloc[:val_split_idx]
X_val = train_data[features].iloc[val_split_idx:]
y_train = train_data[target].iloc[:val_split_idx]
y_val = train_data[target].iloc[val_split_idx:]

# --- Scaling ---
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_data[features])

# --- Model Architecture (Classifier) ---
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# --- Callbacks ---
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, min_delta=0.001)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# --- Training ---
history = model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=8,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# --- Evaluation ---
test_probs = model.predict(X_test_scaled).flatten()
test_preds = (test_probs > 0.5).astype(int)
from sklearn.metrics import classification_report
print("\nTest Classification Report:")
print(classification_report(test_data[target], test_preds, target_names=["Put/Down", "Call/Up"]))

# --- Save Model (TF Lite) ---
model_version = version_base
# Convert directly from in-memory Keras model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
tflite_path = os.path.join(save_dir, f"{model_type}_{model_version}.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

# Optionally, save Keras model for retraining (not needed for TFLite inference)
# keras_model_path = os.path.join(save_dir, f"{model_type}_{model_version}.h5")
# model.save(keras_model_path, include_optimizer=False)

# Save metadata (scaler, features, etc.)
metadata = {
    'scaler': scaler,
    'features': features,
    'target': target,
    'thresholds': {
        'up_threshold': up_threshold,
        'down_threshold': down_threshold
    },
    'metadata': {
        'model_version': model_version,
        'model_type': model_type,
        'ticker': 'SPY',
        'start_date': start_date,
    },
}
joblib.dump(metadata, os.path.join(save_dir, f"{model_type}_{model_version}_metadata.pkl"))

print(f"Model and metadata saved to {save_dir} as {model_type}_{model_version}")
