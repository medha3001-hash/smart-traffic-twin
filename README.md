# 🚦 Smart Traffic Light Digital Twin

A machine learning-powered digital twin that predicts traffic congestion 
and optimizes signal timing at 4 junctions in real time.

Built from scratch using Python, Scikit-learn, and Streamlit.

🌐 **Live Demo:** https://smart-traffic-digital-twin.streamlit.app

---

## 📌 Project Overview

This project simulates a smart traffic control system using real traffic 
data. It uses machine learning to predict congestion and automatically 
adjusts traffic light green times based on ML congestion probability.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Dataset | 48,120 real traffic records |
| Junctions | 4 |
| Time period | 2015 – 2017 |
| ML Model | Random Forest Classifier |
| Model Accuracy | 77.7% |
| Baseline Accuracy | 40.7% |
| Improvement over baseline | +37.1% |
| Data Leakage | None |
| Train-Test Split | Chronological (80/20) |

---

## 🎯 Features

### Core
- ✅ Real traffic dataset — 48,120 records across 4 junctions
- ✅ Random Forest ML model with honest 77.7% accuracy
- ✅ Zero data leakage — label and features computed separately
- ✅ Chronological train-test split — train on past, test on future
- ✅ ML congestion probability directly controls signal cycle time
- ✅ Historical simulation — replays 2 years of real data in real time

### Dashboard
- 🚨 Peak hour indicator — detects morning (8-10am) and evening (5-8pm) rush
- 📊 ML confidence display — shows prediction confidence as a progress bar
- 🔮 What-if analysis — simulates +10 vehicles and shows impact
- 💡 Smart recommendations — text advice based on ML output
- 🎨 Color-coded output — 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW congestion
- 📈 Historical trend graphs — by hour and over calendar time

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.11 | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Random Forest ML model |
| Streamlit | Interactive web dashboard |
| Pickle | Saving and loading trained model |

---

## 📁 Project Structure
```
smart-traffic-twin/
├── data/
│   ├── traffic.csv               # Raw Kaggle dataset
│   ├── traffic_model.pkl         # Trained ML model
│   ├── model_stats.pkl           # Accuracy metrics
│   ├── hourly_avg.csv            # Historical hour averages
│   ├── junction_avg.csv          # Historical junction averages
│   └── hourday_avg.csv           # Hour + day averages
├── explore_data.py               # Phase 2 — data exploration
├── analyze_data.py               # Phase 3 — data analysis
├── train_model.py                # Phase 4 — ML training
├── optimize_signals.py           # Phase 5 — signal optimization
├── dashboard.py                  # Phase 6 — Streamlit dashboard
├── requirements.txt              # All required libraries
└── README.md                     # This file
```

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/medha3001-hash/smart-traffic-twin.git
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

**4. Train the model**
```bash
python3 train_model.py
```

**5. Launch the dashboard**
```bash
streamlit run dashboard.py
```

---

## 🤖 How the ML Works

### Features used (no current vehicle counts — avoids leakage)
| Feature | Meaning |
|---------|---------|
| hour | Hour of day (0–23) |
| day_of_week | 0=Monday to 6=Sunday |
| month | Month of year |
| is_weekend | 1 if Saturday/Sunday |
| hist_hour_avg | Historical avg vehicles at this hour |
| hist_hourday_avg | Historical avg at this hour + day combo |
| hist_junction_avg | Historical avg at this junction |
| hour_vs_day_ratio | Is this hour busier than the daily average? |


### How ML drives signal timing
```
Congestion probability > 0.6  →  160s total cycle (EXTENDED)
Congestion probability > 0.3  →  120s total cycle (NORMAL)
Congestion probability > 0.15 →  100s total cycle (MODERATE)
Congestion probability < 0.15 →   80s total cycle (REDUCED)

Green time per junction = (vehicles / total) × cycle time
```

---

## ⚠️ Known Limitations

- Prediction uses historical time patterns only — not live vehicle counts
- No weather, holiday, or incident data included
- Signal distribution is proportional math — not true optimization
- No inter-junction interaction modelling
- Dataset is from 2015–2017 — patterns may have shifted

---

## 📄 Dataset

**Traffic Flow Forecasting Dataset**  
Source: [Kaggle](https://www.kaggle.com/datasets/fedesoriano/traffic-flow-forecasting-dataset)  
Records: 48,120 hourly vehicle counts  
Period: November 2015 – June 2017  
Junctions: 4

---

## 👩‍💻 Author

**Medha Bhardwaj**  
Built as a complete end-to-end ML project — from raw data to 
interactive dashboard 

- GitHub: [medha3001-hash](https://github.com/medha3001-hash)

---

## 📄 License

This project is open source and available under the MIT License.