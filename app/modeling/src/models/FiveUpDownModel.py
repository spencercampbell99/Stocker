from models.DataHandler import get_up_down_percent_model_data
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report
import pandas as pd
import tensorflow as tf

start_date = "2017-01-01"
up_threshold = 1.005
down_threshold = 0.995

# Fetch and prepare data
final_data = get_up_down_percent_model_data(
    start_date=start_date,
    ticker="SPY",
    up_threshold=up_threshold,
    down_threshold=down_threshold
)

# Corrected features list (removed duplicate)
features = [
    'premarket_pct_change',
    'last_pm_9ma_diff',
    'last_pm_20ma_diff',
    'daily_9ma_diff',
    'daily_20ma_diff',
    'daily_ma9_slope',
    'daily_ma20_slope',
    '5min_premarket_9ma_slope',
    '5min_premarket_20ma_slope',
]
target = 'move_status'

# Time-based split
split_date = pd.to_datetime("2022-06-01").date()
train_data = final_data[final_data.index < split_date]
test_data = final_data[final_data.index >= split_date]

# Time-based validation split (last 10% of training period)
val_split_idx = int(len(train_data) * 0.9)
X_train = train_data[features].iloc[:val_split_idx]
X_val = train_data[features].iloc[val_split_idx:]
y_train = train_data[target].iloc[:val_split_idx]
y_val = train_data[target].iloc[val_split_idx:]

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_data[features])

# Calculate class weights
class_counts = y_train.value_counts()
class_weights = {i: len(y_train)/(3 * count) for i, count in enumerate(class_counts)}

# Improved model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Custom learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Enhanced callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    min_delta=0.001
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

# Train model
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=8,
    validation_data=(X_val_scaled, y_val),
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# Evaluate on test data
test_predictions = model.predict(X_test_scaled)
test_pred_labels = tf.argmax(test_predictions, axis=1).numpy()

# Generate comprehensive evaluation report
print("\nTest Set Evaluation:")
print(classification_report(test_data[target], test_pred_labels))

# Detailed predictions output
print("\nSample Predictions vs Actual:")
for i in range(len(test_pred_labels)):
    print(f"Date: {test_data.index[i]}, "
          f"Predicted: {test_pred_labels[i]}, "
          f"Actual: {test_data[target].iloc[i]}")

print("2025 Accuracy: ", (test_pred_labels == test_data[target].values).mean())

# Use Joblib to save model to saved_models
save = True

if save:
    import joblib
    model_version = "v0.2"
    model.save(f"saved_models/TfUpDownModel_{model_version}.h5")
    
    additonal_data = {
        'scaler': scaler,
        'features': features,
        'target': target,
        'class_weights': class_weights,
        'thresholds': {
            'up_threshold': up_threshold,
            'down_threshold': down_threshold
        },
        'metadata': {
            'model_version': model_version,
            'model_type': 'TfUpDownModel',
            'ticker': 'SPY',
            'start_date': start_date,
            'trained_through_date': str(split_date),
        },
    }
    
    joblib.dump(additonal_data, f"saved_models/TfUpDownModel_{model_version}_metadata.pkl")