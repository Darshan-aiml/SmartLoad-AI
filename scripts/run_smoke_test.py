"""Quick smoke test for baseline pipeline.
Loads a small portion of the data (or cleaned fallback), runs a quick train/test split,
fits LinearRegression and a small RandomForest, and prints RMSE/MAE.
This intentionally uses small models/datasets to finish fast.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# locate data
candidates = [
    os.path.join('..', 'data', 'processed', 'hourly_power_data.csv'),
    os.path.join('..', 'data', 'cleaned_energy_data.csv'),
    os.path.join('..', 'data', 'household_power_consumption.txt'),
    os.path.join('data', 'processed', 'hourly_power_data.csv'),
    os.path.join('data', 'cleaned_energy_data.csv'),
    os.path.join('data', 'household_power_consumption.txt'),
]

data_path = None
for c in candidates:
    if os.path.exists(c):
        data_path = c
        break

if data_path is None:
    print('No data file found. Looked for:')
    for c in candidates:
        print('  -', c)
    sys.exit(2)

print('Using data:', data_path)
# read with fallback options
if data_path.endswith('.txt'):
    try:
        data = pd.read_csv(data_path, sep=';', parse_dates=['Datetime'], index_col='Datetime', low_memory=False)
    except Exception:
        data = pd.read_csv(data_path, parse_dates=['Datetime'], index_col='Datetime', low_memory=False)
else:
    data = pd.read_csv(data_path, parse_dates=['Datetime'], index_col='Datetime')

# quick subsample to keep runtime small
max_rows = 5000
if len(data) > max_rows:
    df = data.iloc[:max_rows].copy()
else:
    df = data.copy()

# basic features
if 'Datetime' in df.columns:
    df.index = pd.to_datetime(df['Datetime'])

if not isinstance(df.index, pd.DatetimeIndex):
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

if not isinstance(df.index, pd.DatetimeIndex):
    print('Data index is not datetime-like; aborting smoke test')
    sys.exit(3)

# feature engineering
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek

target = 'Global_active_power'
if target not in df.columns:
    print(f'Missing target column "{target}" in data; available columns: {list(df.columns)}')
    sys.exit(4)

# drop NA target
df = df.dropna(subset=[target])

# features: time features + numeric columns except target
X = df[['hour', 'dayofweek']].copy()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)
for c in num_cols:
    if c not in X.columns:
        X[c] = df[c]

y = df[target].copy()

# small time-based split
split_idx = int(len(X) * 0.8)
if split_idx < 10:
    print('Not enough rows for train/test split; need at least 10 rows')
    sys.exit(5)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# fit small models
lr = LinearRegression()
lr.fit(X_train_s, y_train)
rf = RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=1)
rf.fit(X_train_s, y_train)

# predict
y_pred_lr = lr.predict(X_test_s)
y_pred_rf = rf.predict(X_test_s)

rmse = lambda a, b: np.sqrt(mean_squared_error(a, b))

print('\nSmoke test results:')
print('LinearRegression RMSE:', rmse(y_test, y_pred_lr), 'MAE:', mean_absolute_error(y_test, y_pred_lr))
print('RandomForest    RMSE:', rmse(y_test, y_pred_rf), 'MAE:', mean_absolute_error(y_test, y_pred_rf))

print('\nSmoke test completed successfully')
