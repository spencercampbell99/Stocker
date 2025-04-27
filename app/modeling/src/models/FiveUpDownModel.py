import tensorflow as tf
from db.database import get_session, DailyCandle
import pandas as pd
from sqlalchemy import text
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Gather data
db = get_session()

# Get daily data back to 2018-01-01
query = db.query(DailyCandle).filter(DailyCandle.ticker == "SPY", DailyCandle.timestamp >= "2018-01-01").order_by(DailyCandle.timestamp.asc())
data = pd.read_sql(query.statement, db.connection())
data = data.set_index("timestamp")
data = data.drop(columns=["id", "ticker"])

# Get 5 minute data for premarket (7am-9:30am) back to 2018-01-01
query = text(f"""
    SELECT
        *
    FROM stocks_fivemincandle
    WHERE ticker = 'SPY'
        AND "timestamp" >= '2018-01-01'
        AND ("timestamp"::time BETWEEN '04:00:00' AND '09:29:00')
    ORDER BY "timestamp" ASC;
""")
data_5min = pd.read_sql(query, db.connection())
data_5min = data_5min.set_index("timestamp")
data_5min = data_5min.drop(columns=["id", "ticker"])

# Get High/Low for each day from hour candle data
from models.SqlQueries import get_daily_move_status_query
query = get_daily_move_status_query()

hl_data = pd.read_sql(query, db.connection())
hl_data['date'] = pd.to_datetime(hl_data['date'])
# Calculate moving averages
data["MA9"] = data["close"].rolling(window=9).mean()
data["MA20"] = data["close"].rolling(window=20).mean()

data["date"] = data.index.date

# Extract date from timestamp index to group by day
data_5min['date'] = data_5min.index.date

data_5min["MA9"] = data_5min.groupby('date')['close'].transform(
    lambda x: x.rolling(window=9).mean()
)
data_5min["MA20"] = data_5min.groupby('date')['close'].transform(
    lambda x: x.rolling(window=20).mean()
)

# Drop NA for moving averages missing
data = data.dropna()
data_5min = data_5min.dropna()

# Calculate slopes for daily data
# Using the difference between last value and value 3 periods back as slope proxy
data['daily_9ma_slope'] = (data['MA9'] - data['MA9'].shift(3)) / 3
data['daily_20ma_slope'] = (data['MA20'] - data['MA20'].shift(3)) / 3

# For 5min data, get the last value before market open (9:30am) each day
last_5min_before_open = data_5min.between_time('09:25', '09:30').groupby('date').last()

# Calculate slopes for 5min premarket data
# Get the slope over the last 30 minutes (6 periods) before market open
def get_slope(group, ma_col, window=6):
    last_values = group[ma_col].tail(window)
    if len(last_values) < window:
        return None
    x = range(window)
    slope = np.polyfit(x, last_values, 1)[0]
    return slope

# Calculate slopes for each day's premarket data
pm_slopes = data_5min.groupby('date').apply(
    lambda x: pd.Series({
        '5min_premarket_9ma_slope': get_slope(x, 'MA9'),
        '5min_premarket_20ma_slope': get_slope(x, 'MA20')
    })
)

# Combine all data
final_data = pd.DataFrame(index=data.index)
final_data['date'] = data['date']
final_data['daily_9ma_slope'] = data['daily_9ma_slope']
final_data['daily_20ma_slope'] = data['daily_20ma_slope']
final_data['close'] = data['close']

# Merge with premarket slopes
final_data = final_data.merge(pm_slopes, on='date', how='left')

final_data = final_data.dropna()

hl_data['date'] = hl_data['date'].dt.date

# Merge with our final_data
final_data = final_data.merge(hl_data[['date', 'move_status', 'open']], on='date', how='left')

# Drop rows where move_status is NA
final_data = final_data.dropna(subset=['move_status'])

# Make sure index is datetime (should probably clean this up better than doing all these conversions but its late and im tired)
final_data['date'] = pd.to_datetime(final_data['date'])
final_data = final_data.set_index('date')

# calculate premarket % change from previous daily close
final_data['premarket_pct_change'] = (final_data['open'] - final_data['close'].shift(1)) / final_data['close'].shift(1) * 100

# Split to final_data train for up to 2025-01-01 and test for 2025-01-01 to 2025-12-31
train_data = final_data[final_data.index < '2025-01-01']
test_data = final_data[final_data.index >= '2025-01-01']

features = ['5min_premarket_9ma_slope', '5min_premarket_20ma_slope', 'daily_20ma_slope', 'premarket_pct_change']
target = 'move_status'

# Prepare features and target
X = train_data[features].values
y = train_data[target].astype(int)  # Convert to integer classes

# Check class distribution
print("Class distribution:")
print(y.value_counts())

# Split into train and test sets
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build a classification model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: 0, 1, 2 (Down, Neutral, Up)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Add class weights to handle imbalance
class_counts = y_train.value_counts()
total = class_counts.sum()
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}

# Train the model
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ],
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Make predictions
predictions = model.predict(X_test_scaled)
predicted_classes = np.argmax(predictions, axis=1)

# Classification report
# print("\nClassification Report:")
# print(classification_report(y_test, predicted_classes, target_names=['Down', 'Neutral', 'Up']))

# # Confusion matrix
# cm = confusion_matrix(y_test, predicted_classes)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#             xticklabels=['Down', 'Neutral', 'Up'],
#             yticklabels=['Down', 'Neutral', 'Up'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# Evaluate the model on the 2025 test set
X_test_2025 = test_data[features].values
y_test_2025 = test_data[target].astype(int)

X_test_2025_scaled = scaler.transform(X_test_2025)
predictions_2025 = model.predict(X_test_2025_scaled)
predicted_classes_2025 = np.argmax(predictions_2025, axis=1)

# print results for 2025
# for i in range(len(predicted_classes_2025)):
#     print(f"Date: {test_data.index[i].date()}, Actual: {y_test_2025.iloc[i]}, Predicted: {predicted_classes_2025[i]}")

# plot predictions for each day of 2025 vs actual move status
# plt.figure(figsize=(14, 7))
# plt.plot(test_data.index, test_data['move_status'], label='Actual Move Status', color='blue', alpha=0.5)
# plt.plot(test_data.index, predicted_classes_2025, label='Predicted Move Status', color='red', alpha=0.5)
# plt.title('Predicted vs Actual Move Status for 2025')
# plt.xlabel('Date')
# plt.ylabel('Move Status')
# plt.legend()
# plt.show()

# print accuracy for 2025
accuracy_2025 = np.mean(predicted_classes_2025 == y_test_2025)
print(f"\nTest Accuracy for 2025: {accuracy_2025:.4f}")

# Train xgboost model
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare features and target for XGBoost
X = final_data[features]
y = final_data[target].astype(int)  # Convert to integer classes

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")

# Test against 2025
X_test_2025 = test_data[features]
y_test_2025 = test_data[target].astype(int)

X_test_2025_scaled = scaler.transform(X_test_2025)
predictions_2025 = xgb_model.predict(X_test_2025_scaled)
accuracy_2025 = accuracy_score(y_test_2025, predictions_2025)

print("XGBoost Model Predictions for 2025:")
for i in range(len(predictions_2025)):
    print(f"Date: {test_data.index[i].date()}, Actual: {y_test_2025.iloc[i]}, Predicted: {predictions_2025[i]}")
    
print(f"\nXGBoost Test Accuracy for 2025: {accuracy_2025:.4f}")