import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# load dataset
df_path = "/Users/kyleschmoyer/Downloads/cs229project/updatedinfo.csv"
df = pd.read_csv(df_path)

# convert datetime column to datetime format and sort by date
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values(by='datetime')

# fill missing values using backward fill
df['volatility'] = df['volatility'].bfill()
df['volatility_10day_ma'] = df['volatility_10day_ma'].bfill()

# feature engineering
df['price_range'] = df['high'] - df['low']
df['price_change'] = (df['close'] - df['open']) / df['open']
df['price_change_lag_1'] = df['price_change'].shift(1)
df['price_change_mean_5'] = df['price_change'].rolling(window=5).mean()
df['price_change_std_5'] = df['price_change'].rolling(window=5).std()
df['rolling_mean_5'] = df['close'].rolling(window=5).mean().shift(1)
df['rolling_mean_20'] = df['close'].rolling(window=20).mean()
df['rolling_mean_ratio'] = df['rolling_mean_5'] / df['rolling_mean_20']
df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
df['cumulative_return'] = df['price_change'].cumsum()

# drop rows with missing values after feature engineering
df = df.dropna()

# define features and target variable
X = df.drop(columns=['price_change', 'open', 'datetime', 'Ticker', 'price_range', 'Filed', 'high', 'low', 'close'])
y = df['price_change']

# split the dataset into train and test sets, reserving the last year for testing
last_year = df['datetime'].max() - pd.DateOffset(years=1)
train_data = df[df['datetime'] < last_year]
test_data = df[df['datetime'] >= last_year]

X_train = train_data.drop(columns=['price_change', 'open', 'datetime', 'Ticker', 'price_range', 'Filed', 'high', 'low', 'close'])
y_train = train_data['price_change']
X_test = test_data.drop(columns=['price_change', 'open', 'datetime', 'Ticker', 'price_range', 'Filed', 'high', 'low', 'close'])
y_test = test_data['price_change']

# standardize the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# train the xgboost regression model with specified hyperparameters
model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8
)
model.fit(X_train_scaled, y_train)

# make predictions on the test set
y_pred = model.predict(X_test_scaled)

# evaluate the model using regression metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"r^2 score: {r2}")
print(f"mean absolute error: {mae}")
print(f"mean squared error: {mse}")
print(f"root mean squared error: {rmse}")

# calculate feature importance and sort it in descending order
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("feature importance:")
print(feature_importance)

# visualize actual vs predicted price changes
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='actual price change', marker='o')
plt.plot(range(len(y_test)), y_pred, label='predicted price change', marker='x')
plt.title('actual vs predicted price change (testing set)')
plt.xlabel('index')
plt.ylabel('price change (%)')
plt.legend()
plt.show()

# plot the distribution of actual vs predicted price change
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=50, alpha=0.6, label="actual")
plt.hist(y_pred, bins=50, alpha=0.6, label="predicted")
plt.title("distribution of actual vs predicted price change")
plt.xlabel("price change (%)")
plt.ylabel("frequency")
plt.legend()
plt.show()
