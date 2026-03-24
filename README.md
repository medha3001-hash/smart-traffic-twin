# 🚦 Smart Traffic Light Digital Twin

A machine learning-powered digital twin that predicts traffic congestion 
and optimizes signal timing at 4 junctions in real time.

## 📌 Project Overview

This project simulates a smart traffic control system using real traffic 
data. It uses machine learning to predict congestion and automatically 
adjusts traffic light green times based on vehicle counts.

## 🎯 Features

- Real traffic data from 4 junctions (48,120 records)
- Traffic pattern analysis with visualizations
- Machine learning model with 97.93% accuracy
- Smart signal timing optimization
- Interactive live dashboard built with Streamlit

## 🛠️ Tech Stack

- **Python 3.11**
- **Pandas** — data loading and manipulation
- **Matplotlib & Seaborn** — data visualization
- **Scikit-learn** — Random Forest machine learning model
- **Streamlit** — interactive web dashboard

## 📁 Project Structure
```
smart-traffic-twin/
├── data/
│   ├── traffic.csv          # Raw traffic dataset
│   └── traffic_model.pkl    # Trained ML model
├── explore_data.py          # Phase 2: Load and understand data
├── analyze_data.py          # Phase 3: Traffic pattern analysis
├── train_model.py           # Phase 4: Train ML model
├── optimize_signals.py      # Phase 5: Signal optimization logic
├── dashboard.py             # Phase 6: Streamlit dashboard
├── requirements.txt         # Required Python libraries
└── README.md                # Project documentation
```

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/smart-traffic-twin.git
cd smart-traffic-twin
```

**2. Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install libraries**
```bash
pip install -r requirements.txt
```

**4. Run the dashboard**
```bash
streamlit run dashboard.py
```

## 📊 Results

| Metric | Value |
|--------|-------|
| Dataset size | 48,120 records |
| Junctions | 4 |
| Model | Random Forest |
| Accuracy | 77.7% |
| Dashboard | Live interactive |

## 🧠 How It Works

1. **Data** — Real hourly vehicle counts at 4 junctions (2015-2017)
2. **Analysis** — Traffic patterns by hour, day, and junction
3. **ML Model** — Random Forest classifies each period as congested or not
4. **Optimization** — Green light duration assigned proportionally to vehicle count
5. **Dashboard** — Streamlit app shows everything interactively

## 👩‍💻 Author

**Medha Bhardwaj**  
Built as a beginner ML project from scratch.

## 📄 Dataset

Traffic Flow Forecasting Dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/traffic-flow-forecasting-dataset)