import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset
df = pd.read_csv("/Users/kyleschmoyer/Downloads/cs229project/updatedinfo.csv")

# Convert datetime and sort
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')

# Fill missing values
df.fillna(method='bfill', inplace=True)

# Feature engineering
df['price_range'] = df['high'] - df['low']
df['price_change'] = (df['close'] - df['open']) / df['open']
df['rolling_mean_5'] = df['close'].rolling(window=5).mean().shift(1)
df['rolling_mean_20'] = df['close'].rolling(window=20).mean()
df['rolling_mean_ratio'] = df['rolling_mean_5'] / df['rolling_mean_20']
df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

# Drop rows with missing values
df = df.dropna()

# Define features and target
features = ['volatility', 'volatility_10day_ma', 'price_change_lag_1', 
            'rolling_mean_ratio', 'vwap', 'cumulative_return']
X = df[features]
y = df['price_change']

# Split into train and test sets
last_year = df['datetime'].max() - pd.DateOffset(years=1)
train_data = df[df['datetime'] < last_year]
test_data = df[df['datetime'] >= last_year]

X_train = train_data[features]
y_train = train_data['price_change']
X_test = test_data[features]
y_test = test_data['price_change']

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)  # Output layer
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Train model
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, 
                    validation_data=(X_test_reshaped, y_test), verbose=1)

# Predictions
y_pred = model.predict(X_test_reshaped)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Actual Price Change', marker='o')
plt.plot(range(len(y_test)), y_pred, label='Predicted Price Change', marker='x')
plt.title('Actual vs Predicted Price Change')
plt.xlabel('Index')
plt.ylabel('Price Change')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
