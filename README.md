# SmartLoad-AI
# ⚡ AI-Based Smart Home Energy Load Balancer


---

## 🧭 Overview

The **AI-Based Smart Home Energy Load Balancer** is a machine learning and IoT-integrated system designed to **predict household energy consumption** and **balance load automatically** to prevent circuit overloads and improve energy efficiency.  

By forecasting energy demand and intelligently scheduling appliances, this project contributes to the **sustainable use of electricity** in smart homes — reducing waste, preventing power surges, and optimizing daily energy patterns.

---

## 🎯 Problem Statement

In most households:
- ⚡ Multiple high-power appliances (AC, geyser, washing machine) may run simultaneously.  
- 🚨 This often leads to circuit overloads, fuse trips, and inefficient energy usage.  
- 💡 Manual load management is reactive and unreliable.

**Root Cause:** Lack of an intelligent system that predicts and prevents energy overloads in advance.

---

## 💡 Objective

Develop an **AI-powered predictive system** that:
1. Forecasts near-future home energy consumption using **Machine Learning/Deep Learning models**.  
2. Suggests or automatically schedules appliances to **balance load dynamically**.  
3. Integrates with **IoT relays or smart plugs** (real or simulated).  

---

## ⚙️ Technical Approach

| Component | Description |
|------------|--------------|
| **Dataset** | Public smart home energy datasets (e.g., UCI, UK-DALE, REFIT) |
| **Preprocessing** | Data cleaning, normalization, time-series feature engineering |
| **Modeling** | Regression and LSTM models for energy demand forecasting |
| **IoT Simulation** | Raspberry Pi + relays (or simulated control logic) |
| **Evaluation** | RMSE, MAE, and R² Score |
| **Visualization** | Real-time charts using Plotly / Matplotlib |

---
## 🧾 Week 1 Deliverables

- ✅ Created **public GitHub repository**  
- ✅ Collected suitable **energy consumption dataset**  
- ✅ Performed **data preprocessing** in Jupyter Notebook  
- ✅ Uploaded notebook and documentation  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Darshan-aiml/SmartLoad-AI.git
cd SmartLoad-AI
```

2. **Create a virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Dashboard

Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Features of the Dashboard

- 📊 **Overview**: View key metrics and dataset statistics
- 📈 **Data Exploration**: Interactive time-series plots, correlation heatmaps, and sub-metering analysis
- 🤖 **Model Predictions**: Compare Linear Regression and Random Forest model performance
- 🔍 **Feature Analysis**: Examine feature importance and distributions

### Running the Jupyter Notebooks

To explore the development notebooks:
```bash
jupyter lab
```

Then navigate to the `notebooks/` directory to access:
- `01_data_exploration.ipynb` - Data analysis and visualization
- `02_data_preprocessing.ipynb` - Data cleaning and feature engineering
- `03_baseline_model.ipynb` - Model training and evaluation

---

## 🌍 Impact

- ♻️ Promotes **energy-efficient living**.  
- ⚡ Prevents **circuit overloads** and **electrical hazards**.  
- 🧠 Encourages **data-driven power management** at home.  
- 🏡 Can be scaled for **community-level or smart grid integration**.

---

## 🧮 Evaluation Metrics

| Metric | Purpose |
|---------|----------|
| **RMSE** | Measures prediction accuracy of power load |
| **MAE** | Evaluates average error magnitude |
| **R² Score** | Determines model performance in variance explanation |

---

## 🔧 Future Enhancements

- Integrate **real-time IoT data** from sensors.  
- Develop a **mobile app or dashboard** for user interaction.  
- Implement **reinforcement learning** for adaptive load control.  
- Expand to **multi-home neighborhood load prediction**.

---

## 👤 Author

**Dharshan R**  
🎓 AI & ML Enthusiast | Research Intern  


