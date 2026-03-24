import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ============================================================
# LOAD MODEL AND HISTORICAL DATA
# ============================================================
with open('data/traffic_model.pkl', 'rb') as f:
    model = pickle.load(f)

hour_avg_df    = pd.read_csv('data/hourly_avg.csv', index_col=0)['Vehicles']
junc_avg_df    = pd.read_csv('data/junction_avg.csv', index_col=0)['Vehicles']
hourday_avg_df = pd.read_csv('data/hourday_avg.csv')

print("✅ Model and historical data loaded!")

# ============================================================
# HELPER — build the 8 features the model expects
# ============================================================
def build_time_features(hour, day_of_week, j1=None, j2=None,
                        j3=None, j4=None):
    hist_hour = hour_avg.get(hour, hour_avg.mean())

    match = hourday_avg[
        (hourday_avg['hour'] == hour) &
        (hourday_avg['day_of_week'] == day_of_week)
    ]['Vehicles'].values
    hist_hourday = match[0] if len(match) > 0 else hist_hour

    hist_junc  = junc_avg.mean()
    day_avg    = hour_avg.mean()
    hour_ratio = hist_hour / day_avg if day_avg > 0 else 1.0
    is_weekend = 1 if day_of_week >= 5 else 0

    # Calculate current vs historical ratio
    # If vehicle counts provided, use their average
    # Otherwise fall back to 1.0 (exactly average)
    if j1 is not None:
        current_avg = (j1 + j2 + j3 + j4) / 4
        vehicles_vs_hist = current_avg / hist_hour if hist_hour > 0 else 1.0
    else:
        vehicles_vs_hist = 1.0

    return [[
        hour, day_of_week, 6, is_weekend,
        hist_hour, hist_hourday, hist_junc,
        hour_ratio, vehicles_vs_hist
    ]]

# ============================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================
def optimize_signal_timing(j1, j2, j3, j4, hour, day_of_week):
    """
    Step 1: Ask ML model — how likely is congestion right now?
    Step 2: Use that probability to set total cycle time.
    Step 3: Distribute green time across junctions by vehicle count.

    This way ML actually DRIVES the signal behavior.
    """

    # --- Step 1: ML prediction using time + history features ---
    time_features = build_time_features(hour, day_of_week, j1, j2, j3, j4)
    congestion_pred  = model.predict(time_features)[0]
    congestion_proba = model.predict_proba(time_features)[0][1]

    # --- Step 2: ML probability sets total cycle time ---
    if congestion_proba > 0.6:
        total_cycle_time = 160
        cycle_label = "EXTENDED"
    elif congestion_proba > 0.3:
        total_cycle_time = 120
        cycle_label = "NORMAL"
    elif congestion_proba > 0.15:
        total_cycle_time = 100
        cycle_label = "MODERATE"
    else:
        total_cycle_time = 80
        cycle_label = "REDUCED"

    # --- Step 3: Distribute green time by vehicle count ---
    counts        = [j1, j2, j3, j4]
    total_vehicles = sum(counts)

    if total_vehicles == 0:
        weights = [0.25, 0.25, 0.25, 0.25]
    else:
        weights = [c / total_vehicles for c in counts]

    green_times = [
        max(10, min(60, round(w * total_cycle_time)))
        for w in weights
    ]

    # --- Step 4: Label each junction congestion level ---
    def get_level(c):
        if c > 50:    return 'HIGH'
        elif c >= 20: return 'MEDIUM'
        else:         return 'LOW'

    return {
        'counts':           counts,
        'levels':           [get_level(c) for c in counts],
        'green_times':      green_times,
        'congestion_pred':  congestion_pred,
        'congestion_proba': round(congestion_proba * 100, 1),
        'total_cycle_time': total_cycle_time,
        'cycle_label':      cycle_label
    }