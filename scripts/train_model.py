import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import os

# Set up paths
DATA_PATH = 'data/cleaned_energy_data.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

def load_and_prepare_data():
    """Load data and create features"""
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Please run prepare_data.py first.")
        
    df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
    
    # Feature engineering
    print("Feature engineering...")
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['quarter'] = df.index.quarter
    
    target = 'Global_active_power'
    df = df.dropna(subset=[target])
    
    # Select features
    # Using time-based features and potentially other numerical columns if available/appropriate
    # For this baseline, we focus on time features to predict power
    features = ['hour', 'dayofweek', 'month', 'dayofyear', 'quarter']
    
    # Add lag features if possible (simple approach involves just dropping NaNs created)
    # For simplicity in this script, we stick to time-based features + other sensors
    # Check what other numeric columns we have that can be used as features (lagged or concurrent)
    # Caution: Using concurrent sub-metering to predict global active power is a bit of a leak if we want to forecast ahead
    # But for analyzing relationships, it's fine. 
    # Let's stick to time features + numeric measurements ( voltage, intensity etc) to match app.py logic
    
    X = df[features].copy()
    
    # Add other numeric columns except target and sub-metering (to avoid data leakage if predicting total load)
    # Actually, app.py uses ALL numeric columns. Let's inspect what app.py does:
    # app.py matches: X = df[['hour', 'dayofweek', 'month']].copy() + other num cols
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)
    
    # We should exclude sub_metering if the goal is pure forecasting from time, 
    # but the user asked for "Random Forest ML model for prediction and *feature analysis*".
    # Including everything allows seeing which sensors correlate most.
    for c in num_cols:
        if c not in X.columns:
            X[c] = df[c]
            
    y = df[target].copy()
    
    return X, y

def train():
    X, y = load_and_prepare_data()
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Linear Regression
    print("Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    lr_pred = lr.predict(X_test_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    print(f"Linear Regression RMSE: {lr_rmse:.4f}")
    
    # Train Random Forest
    print("Training Random Forest (this may take a moment)...")
    # Using slightly more robust parameters than app.py
    rf = RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    rf_pred = rf.predict(X_test_scaled)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    print(f"Random Forest RMSE: {rf_rmse:.4f}")
    
    # Feature Importance
    print("\nFeature Analysis:")
    importances = rf.feature_importances_
    feature_names = X.columns
    feature_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_imp = feature_imp.sort_values('Importance', ascending=False)
    print(feature_imp.head(10))
    
    # Save models
    print("\nSaving models...")
    joblib.dump(lr, os.path.join(MODELS_DIR, 'linear_regression.joblib'))
    joblib.dump(rf, os.path.join(MODELS_DIR, 'random_forest.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    # Save feature names for later use
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
    
    print(f"Models saved to {MODELS_DIR}/")

if __name__ == "__main__":
    train()
