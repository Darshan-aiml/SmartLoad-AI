"""
SmartLoad-AI: Energy Demand Prediction Dashboard
A Streamlit application for visualizing energy consumption patterns and making predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib

# Page configuration
st.set_page_config(
    page_title="SmartLoad-AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚡ SmartLoad-AI Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Energy Demand Prediction & Analysis Platform")

# Sidebar
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("---")

# Data loading function with caching
@st.cache_data
def load_data():
    """Load and cache the energy data"""
    data_path = 'data/cleaned_energy_data.csv'
    if os.path.exists(data_path):
        data = pd.read_csv(data_path, parse_dates=['Datetime'], index_col='Datetime')
        return data
    else:
        st.error(f"Data file not found at {data_path}")
        return None

# Model training function with caching
@st.cache_resource
def load_models():
    """Load pre-trained models or train if not found"""
    models_dir = 'models'
    try:
        lr = joblib.load(os.path.join(models_dir, 'linear_regression.joblib'))
        rf = joblib.load(os.path.join(models_dir, 'random_forest.joblib'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        return lr, rf, scaler
    except Exception as e:
        st.warning(f"Could not load pre-trained models: {e}. Training on the fly...")
        return None

@st.cache_resource
def train_models(X_train, y_train):
    """Train and cache the baseline models"""
    # Check if models can be loaded first
    models = load_models()
    if models is not None:
        return models

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    
    return lr, rf, scaler

# Load data
data = load_data()

if data is not None:
    # Sidebar options
    page = st.sidebar.selectbox(
        "Select Page",
        ["📊 Overview", "📈 Data Exploration", "🤖 Model Predictions", "🔍 Feature Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Info")
    st.sidebar.info(f"""
    **Records:** {len(data):,}  
    **Features:** {len(data.columns)}  
    **Date Range:** {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}
    """)
    
    # Main content based on page selection
    if page == "📊 Overview":
        st.header("Dashboard Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Average Power (kW)",
                value=f"{data['Global_active_power'].mean():.2f}",
                delta=f"{data['Global_active_power'].std():.2f} std"
            )
        
        with col2:
            st.metric(
                label="Max Power (kW)",
                value=f"{data['Global_active_power'].max():.2f}",
                delta="Peak"
            )
        
        with col3:
            st.metric(
                label="Min Power (kW)",
                value=f"{data['Global_active_power'].min():.2f}",
                delta="Lowest"
            )
        
        with col4:
            st.metric(
                label="Total Records",
                value=f"{len(data):,}",
                delta="Available"
            )
        
        st.markdown("---")
        
        # Recent data preview
        st.subheader("📋 Recent Data Preview")
        st.dataframe(data.tail(10), use_container_width=True)
        
        # Quick stats
        st.subheader("📊 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
        
    elif page == "📈 Data Exploration":
        st.header("Data Exploration & Visualization")
        
        # Time series plot
        st.subheader("🌡️ Energy Consumption Over Time")
        
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", data.index.min())
        with col2:
            end_date = st.date_input("End Date", data.index.max())
        
        # Filter data
        mask = (data.index >= pd.Timestamp(start_date)) & (data.index <= pd.Timestamp(end_date))
        filtered_data = data.loc[mask]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(filtered_data.index, filtered_data['Global_active_power'], linewidth=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Global Active Power (kW)')
        ax.set_title('Energy Demand Over Time')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Sub-metering comparison
        if {'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'}.issubset(set(data.columns)):
            st.subheader("🧩 Sub-Metering Analysis")
            
            sample_size = st.slider("Sample Size", 100, 1000, 500)
            sample_data = filtered_data.head(sample_size)
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(sample_data.index, sample_data['Sub_metering_1'], label='Kitchen', alpha=0.8)
            ax.plot(sample_data.index, sample_data['Sub_metering_2'], label='Laundry', alpha=0.8)
            ax.plot(sample_data.index, sample_data['Sub_metering_3'], label='HVAC', alpha=0.8)
            ax.set_xlabel('Time')
            ax.set_ylabel('Energy (Wh)')
            ax.set_title('Energy Consumption by Sub-Meters')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
    elif page == "🤖 Model Predictions":
        st.header("Model Predictions & Performance")
        
        # Prepare data for modeling
        df = data.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        
        target = 'Global_active_power'
        df = df.dropna(subset=[target])
        
        # Feature selection
        features = ['hour', 'dayofweek', 'month', 'dayofyear', 'quarter']
        X = df[features].copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        for c in num_cols:
            if c not in X.columns:
                X[c] = df[c]
        y = df[target].copy()
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train models
        with st.spinner("Training models..."):
            lr, rf, scaler = train_models(X_train, y_train)
        
        # Make predictions
        X_test_scaled = scaler.transform(X_test)
        y_pred_lr = lr.predict(X_test_scaled)
        y_pred_rf = rf.predict(X_test_scaled)
        
        # Calculate metrics
        def rmse(a, b):
            return np.sqrt(mean_squared_error(a, b))
        
        lr_rmse = rmse(y_test, y_pred_lr)
        lr_mae = mean_absolute_error(y_test, y_pred_lr)
        rf_rmse = rmse(y_test, y_pred_rf)
        rf_mae = mean_absolute_error(y_test, y_pred_rf)
        
        # Display metrics
        st.subheader("📊 Model Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Linear Regression")
            st.metric("RMSE", f"{lr_rmse:.4f}")
            st.metric("MAE", f"{lr_mae:.4f}")
        
        with col2:
            st.markdown("#### Random Forest")
            st.metric("RMSE", f"{rf_rmse:.4f}")
            st.metric("MAE", f"{rf_mae:.4f}")
        
        st.markdown("---")
        
        # Actual vs Predicted plot
        st.subheader("🎯 Actual vs Predicted Values")
        
        model_choice = st.radio("Select Model", ["Linear Regression", "Random Forest"])
        y_pred = y_pred_lr if model_choice == "Linear Regression" else y_pred_rf
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Global Active Power (kW)')
        ax.set_ylabel('Predicted Global Active Power (kW)')
        ax.set_title(f'Actual vs Predicted - {model_choice}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Prediction time series
        st.subheader("📈 Predictions Over Time")
        
        plot_n = st.slider("Number of points to display", 50, 500, 200)
        
        viz = pd.DataFrame({
            'actual': y_test.iloc[:plot_n].values,
            'predicted': y_pred[:plot_n]
        })
        
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(viz['actual'], label='Actual', color='black', linewidth=1)
        ax.plot(viz['predicted'], label=model_choice, alpha=0.8, linewidth=1)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Global Active Power (kW)')
        ax.set_title(f'Predictions vs Actual (First {plot_n} test samples)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
    elif page == "🔍 Feature Analysis":
        st.header("Feature Importance & Analysis")
        
        # Prepare data for modeling
        df = data.copy()
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['dayofyear'] = df.index.dayofyear
        df['quarter'] = df.index.quarter
        
        target = 'Global_active_power'
        df = df.dropna(subset=[target])
        
        # Feature selection
        features = ['hour', 'dayofweek', 'month', 'dayofyear', 'quarter']
        X = df[features].copy()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        for c in num_cols:
            if c not in X.columns:
                X[c] = df[c]
        y = df[target].copy()
        
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train Random Forest
        with st.spinner("Training Random Forest..."):
            _, rf, scaler = train_models(X_train, y_train)
        
        # Feature importance
        st.subheader("📊 Feature Importance (Random Forest)")
        
        importances = rf.feature_importances_
        features = X.columns
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
            ax.set_title('Feature Importance Scores')
            ax.set_xlabel('Importance')
            st.pyplot(fig)
        
        with col2:
            st.dataframe(importance_df, use_container_width=True)
        
        st.markdown("---")
        
        # Feature distributions
        st.subheader("📈 Feature Distributions")
        
        selected_feature = st.selectbox("Select Feature", data.columns.tolist())
        
        fig, ax = plt.subplots(figsize=(10, 5))
        data[selected_feature].hist(bins=50, ax=ax)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'Distribution of {selected_feature}')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

else:
    st.error("Unable to load data. Please check the data file path.")
    st.info("Expected file location: `data/cleaned_energy_data.csv`")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>SmartLoad-AI Dashboard | Energy Demand Prediction & Analysis Platform</p>
    <p>Built with Streamlit 🎈</p>
</div>
""", unsafe_allow_html=True)
