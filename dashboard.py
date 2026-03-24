import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

st.set_page_config(
    page_title="Smart Traffic Light Digital Twin",
    page_icon="🚦",
    layout="wide"
)

@st.cache_resource
def load_model():
    with open('data/traffic_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_stats():
    with open('data/model_stats.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_historical():
    hour_avg    = pd.read_csv('data/hourly_avg.csv', index_col=0)['Vehicles']
    junc_avg    = pd.read_csv('data/junction_avg.csv', index_col=0)['Vehicles']
    hourday_avg = pd.read_csv('data/hourday_avg.csv')
    return hour_avg, junc_avg, hourday_avg

@st.cache_data
def load_traffic_data():
    df = pd.read_csv('data/traffic.csv')
    df['DateTime']    = pd.to_datetime(df['DateTime'])
    df['hour']        = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['month']       = df['DateTime'].dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    return df

model                           = load_model()
stats                           = load_stats()
hour_avg, junc_avg, hourday_avg = load_historical()
df                              = load_traffic_data()

def get_peak_status(hour):
    if 8 <= hour <= 10:
        return "🚨 Morning Peak Hour", True
    elif 17 <= hour <= 20:
        return "🚨 Evening Peak Hour", True
    else:
        return "🙂 Normal Traffic Hours", False

def get_recommendations(result, hour):
    recs    = []
    counts  = result['counts']
    levels  = result['levels']
    green_times = result['green_times']
    proba   = result['congestion_proba']
    names   = ['Junction 1','Junction 2','Junction 3','Junction 4']
    max_idx = counts.index(max(counts))

    if result['congestion_pred'] == 1:
        recs.append(
            f"🔴 **{names[max_idx]}** has the highest load "
            f"({counts[max_idx]} vehicles) — "
            f"green time extended to {green_times[max_idx]}s"
        )

    for i, (gt, cnt) in enumerate(zip(green_times, counts)):
        if gt <= 10 and cnt > 20:
            recs.append(
                f"⚠️ **{names[i]}** has {cnt} vehicles "
                f"but only 10s green — consider manual override"
            )

    _, is_peak = get_peak_status(hour)
    if is_peak and proba > 50:
        recs.append(
            "🕐 This is a **peak hour** — consider deploying "
            "traffic personnel at high-load junctions"
        )

    if proba < 30:
        recs.append(
            "✅ Traffic is light — **REDUCED cycle mode** active "
            "to minimise waiting time"
        )

    if not recs:
        recs.append(
            "✅ Traffic is flowing normally — no intervention needed"
        )

    return recs

def get_bar_colors(levels):
    color_map = {
        'HIGH':   '#e74c3c',
        'MEDIUM': '#f39c12',
        'LOW':    '#2ecc71',
    }
    return [color_map[l] for l in levels]

def get_level_emoji(level):
    return {'HIGH':'🔴','MEDIUM':'🟡','LOW':'🟢'}[level]

def build_time_features(hour, day_of_week):
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

    return [[
        hour, day_of_week, 6, is_weekend,
        hist_hour, hist_hourday, hist_junc, hour_ratio
    ]]

def optimize_signal_timing(j1, j2, j3, j4, hour, day_of_week):
    time_features    = build_time_features(hour, day_of_week)
    congestion_pred  = model.predict(time_features)[0]
    congestion_proba = model.predict_proba(time_features)[0][1]

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

    counts         = [j1, j2, j3, j4]
    total_vehicles = sum(counts)
    weights        = ([c / total_vehicles for c in counts]
                      if total_vehicles > 0
                      else [0.25, 0.25, 0.25, 0.25])

    green_times = [
        max(10, min(60, round(w * total_cycle_time)))
        for w in weights
    ]

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

# ============================================================
# HEADER
# ============================================================
st.title("🚦 Smart Traffic Light Digital Twin")
st.markdown("### ML-driven congestion prediction and signal optimization")

c1, c2, c3 = st.columns(3)
c1.metric("Model Accuracy",          f"{stats['model_acc']*100:.1f}%")
c2.metric("Baseline Accuracy",       f"{stats['baseline_acc']*100:.1f}%")
c3.metric("Improvement over baseline",f"+{stats['improvement']*100:.1f}%")
st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Control Panel")
run_simulation = st.sidebar.checkbox(
    "▶ Run Historical Simulation",
    help="Replays real historical data step by step"
)
if run_simulation:
    sim_speed = st.sidebar.slider("Speed (seconds per step)", 0.05, 1.0, 0.15)

st.sidebar.markdown("---")
st.sidebar.markdown("**Manual mode controls**")
st.sidebar.caption("Used when simulation is off")

hour        = st.sidebar.slider("Hour of Day", 0, 23, 8)
day_of_week = st.sidebar.selectbox(
    "Day of Week", options=[0,1,2,3,4,5,6],
    format_func=lambda x: [
        'Monday','Tuesday','Wednesday',
        'Thursday','Friday','Saturday','Sunday'][x]
)
j1 = st.sidebar.slider("Junction 1 vehicles", 0, 180, 80)
j2 = st.sidebar.slider("Junction 2 vehicles", 0, 180, 60)
j3 = st.sidebar.slider("Junction 3 vehicles", 0, 180, 40)
j4 = st.sidebar.slider("Junction 4 vehicles", 0, 180, 20)

# ============================================================
# RENDER FUNCTION — all fixes applied, no context manager errors
# ============================================================
def render_dashboard(j1, j2, j3, j4, hour, day_of_week,
                     sim_label=None, placeholders=None):

    result    = optimize_signal_timing(j1, j2, j3, j4, hour, day_of_week)
    use_ph    = placeholders is not None
    proba     = result['congestion_proba']
    bar_cols  = get_bar_colors(result['levels'])
    peak_label, is_peak = get_peak_status(hour)

    # --- Peak hour banner ---
    if use_ph:
        with placeholders['peak'].container():
            if is_peak:
                st.warning(f"**{peak_label}** — historically high traffic period")
            else:
                st.info(f"**{peak_label}**")
    else:
        if is_peak:
            st.warning(f"**{peak_label}** — historically high traffic period")
        else:
            st.info(f"**{peak_label}**")

    # --- Metrics row ---
    if use_ph:
        with placeholders['metrics'].container():
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("Junction 1", f"{j1} vehicles")
            c2.metric("Junction 2", f"{j2} vehicles")
            c3.metric("Junction 3", f"{j3} vehicles")
            c4.metric("Junction 4", f"{j4} vehicles")
            c5.metric("Total",      f"{j1+j2+j3+j4}")
            c6.metric("Hour",       f"{hour}:00")
    else:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Junction 1", f"{j1} vehicles")
        c2.metric("Junction 2", f"{j2} vehicles")
        c3.metric("Junction 3", f"{j3} vehicles")
        c4.metric("Junction 4", f"{j4} vehicles")
        c5.metric("Total",      f"{j1+j2+j3+j4}")
        c6.metric("Hour",       f"{hour}:00")

    # --- ML prediction + signal chart ---
    if use_ph:
        signal_con = placeholders['signals'].container()
    else:
        signal_con = st.container()

    with signal_con:
        left, right = st.columns([1, 2])

        with left:
            st.subheader("🤖 ML Prediction")

            if result['congestion_pred'] == 1:
                conf_label = "HIGH" if proba >= 70 else "MODERATE"
                if proba >= 70:
                    st.error(f"🔴 CONGESTED — **{conf_label}** confidence ({proba}%)")
                else:
                    st.warning(f"🟡 LIKELY CONGESTED — **{conf_label}** confidence ({proba}%)")
            else:
                conf_label = "HIGH" if proba < 30 else "MODERATE"
                if proba < 30:
                    st.success(f"🟢 NORMAL — **{conf_label}** confidence ({100-proba:.1f}%)")
                else:
                    st.info(f"🟡 LIKELY NORMAL — **{conf_label}** confidence ({100-proba:.1f}%)")

            st.markdown("**Congestion probability:**")
            st.progress(int(proba))
            st.caption(f"{proba}% chance of congestion at hour {hour}:00")
            st.markdown(f"**Cycle mode:** `{result['cycle_label']}`")
            st.markdown(f"**Total cycle time:** `{result['total_cycle_time']}s`")
            st.caption(f"ML probability: {proba}% → cycle adjusted accordingly")
            st.markdown("---")
            st.markdown("**Junction breakdown:**")
            for i, (lvl, cnt, gt) in enumerate(zip(
                    result['levels'], result['counts'], result['green_times'])):
                emoji = get_level_emoji(lvl)
                st.write(f"{emoji} **J{i+1}:** {lvl} ({cnt} vehicles) → **{gt}s** green")

        with right:
            st.subheader("🚦 Optimized Signal Timings")
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(
                ['Junction 1','Junction 2','Junction 3','Junction 4'],
                result['green_times'],
                color=bar_cols,
                edgecolor='white', linewidth=1.5
            )
            for bar, val, lvl in zip(bars, result['green_times'], result['levels']):
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f'{val}s\n({lvl})',
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold'
                )
            ax.set_ylabel('Green light duration (seconds)')
            ax.set_title(
                f'Hour {hour}:00 — Cycle: {result["total_cycle_time"]}s '
                f'[{result["cycle_label"]}]'
            )
            ax.set_ylim(0, 80)
            ax.axhline(30, color='gray', linestyle='--',
                       alpha=0.4, label='Average baseline')
            ax.legend(fontsize=8)
            ax.set_facecolor('#f8f9fa')
            fig.patch.set_facecolor('#ffffff')
            st.pyplot(fig)
            plt.close()

    # --- Recommendations ---
    if use_ph:
        rec_con = placeholders['recs'].container()
    else:
        rec_con = st.container()

    with rec_con:
        st.markdown("---")
        st.subheader("💡 Recommendations")
        recs = get_recommendations(result, hour)
        for r in recs:
            st.markdown(f"- {r}")

    # --- What-if analysis ---
    if use_ph:
        whatif_con = placeholders['whatif'].container()
    else:
        whatif_con = st.container()

    with whatif_con:
        st.markdown("---")
        st.subheader("🔮 What-if Analysis")
        st.caption("What happens if traffic increases by +10 vehicles at every junction?")

        wj1, wj2, wj3, wj4 = j1+10, j2+10, j3+10, j4+10
        what_if  = optimize_signal_timing(wj1, wj2, wj3, wj4, hour, day_of_week)
        wi_proba = what_if['congestion_proba']

        wcol1, wcol2, wcol3 = st.columns(3)

        with wcol1:
            st.markdown("**Current**")
            st.markdown(f"Vehicles: {j1} / {j2} / {j3} / {j4}")
            st.markdown(
                f"Prediction: {'🔴 CONGESTED' if result['congestion_pred']==1 else '🟢 NORMAL'}"
            )
            st.markdown(f"Confidence: {proba}%")
            st.markdown(f"Cycle: {result['total_cycle_time']}s [{result['cycle_label']}]")

        with wcol2:
            st.markdown("**After +10 vehicles each**")
            st.markdown(f"Vehicles: {wj1} / {wj2} / {wj3} / {wj4}")
            st.markdown(
                f"Prediction: {'🔴 CONGESTED' if what_if['congestion_pred']==1 else '🟢 NORMAL'}"
            )
            st.markdown(f"Confidence: {wi_proba}%")
            st.markdown(f"Cycle: {what_if['total_cycle_time']}s [{what_if['cycle_label']}]")

        with wcol3:
            st.markdown("**Impact**")
            proba_diff = wi_proba - proba
            cycle_diff = what_if['total_cycle_time'] - result['total_cycle_time']

            if proba_diff > 0:
                st.markdown(f"📈 Congestion risk **+{proba_diff:.1f}%**")
            else:
                st.markdown(f"📉 Congestion risk **{proba_diff:.1f}%**")

            if cycle_diff > 0:
                st.markdown(f"⏱️ Cycle time **+{cycle_diff}s longer**")
            elif cycle_diff < 0:
                st.markdown(f"⏱️ Cycle time **{cycle_diff}s shorter**")
            else:
                st.markdown("⏱️ Cycle time **unchanged**")

            st.markdown("Green time changes:")
            for i, (cur, new) in enumerate(
                    zip(result['green_times'], what_if['green_times'])):
                diff  = new - cur
                arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
                st.caption(
                    f"J{i+1}: {cur}s {arrow} {new}s "
                    f"({'+' if diff>=0 else ''}{diff}s)"
                )

    # --- Historical trend ---
    if use_ph:
        history_con = placeholders['history'].container()
    else:
        history_con = st.container()

    with history_con:
        st.markdown("---")
        st.subheader("📈 Historical Traffic Trend")
        tab1, tab2 = st.tabs(["By Hour of Day", "Over Time (sample)"])

        with tab1:
            hourly = df.groupby(['hour','Junction'])['Vehicles'].mean().reset_index()
            fig2, ax2 = plt.subplots(figsize=(12, 3))
            colors2 = ['#2ecc71','#3498db','#e74c3c','#f39c12']
            for junc, col in zip([1,2,3,4], colors2):
                d = hourly[hourly['Junction'] == junc]
                ax2.plot(d['hour'], d['Vehicles'],
                         marker='o', label=f'Junction {junc}',
                         color=col, linewidth=2, markersize=3)
            ax2.axvspan(8,  10, alpha=0.1, color='red',    label='Morning peak')
            ax2.axvspan(17, 20, alpha=0.1, color='orange', label='Evening peak')
            ax2.axvline(hour, color='purple', linestyle='--',
                        linewidth=2, label=f'Current ({hour}:00)')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Average Vehicles')
            ax2.set_title('Average Vehicle Count by Hour (shaded = peak periods)')
            ax2.legend(fontsize=7, ncol=4)
            ax2.set_xticks(range(0, 24))
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
            plt.close()

        with tab2:
            df_sample = (df[df['Junction'] == 1]
                           .sort_values('DateTime')
                           .iloc[::24].copy())
            fig3, ax3 = plt.subplots(figsize=(12, 3))
            ax3.plot(df_sample['DateTime'], df_sample['Vehicles'],
                     color='#3498db', linewidth=1, alpha=0.8)
            ax3.fill_between(df_sample['DateTime'],
                             df_sample['Vehicles'],
                             alpha=0.15, color='#3498db')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Vehicles (Junction 1)')
            ax3.set_title('Traffic Volume Over Time — Junction 1 (daily sample)')
            ax3.grid(True, alpha=0.3)
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

    if use_ph and sim_label:
        placeholders['siminfo'].info(f"🔄 Simulating: **{sim_label}**")

# ============================================================
# SIMULATION MODE
# ============================================================
if run_simulation:
    st.markdown("### 🔄 Simulation Mode — replaying historical data")
    st.caption("The optimizer processes every real historical hour in sequence")

    df_sim = df.pivot_table(
        index=['DateTime','hour','day_of_week','month','is_weekend'],
        columns='Junction', values='Vehicles'
    ).reset_index().dropna()

    df_sim.columns = ['DateTime','hour','day_of_week',
                      'month','is_weekend','j1','j2','j3','j4']
    df_sim = df_sim.sort_values('DateTime').reset_index(drop=True)
    df_sim = df_sim.iloc[::6].reset_index(drop=True)

    progress_bar = st.progress(0)
    total_steps  = len(df_sim)

    placeholders = {
        'peak':    st.empty(),
        'metrics': st.empty(),
        'signals': st.empty(),
        'recs':    st.empty(),
        'whatif':  st.empty(),
        'history': st.empty(),
        'siminfo': st.empty(),
    }

    for step, (_, row) in enumerate(df_sim.iterrows()):
        render_dashboard(
            j1=int(row['j1']), j2=int(row['j2']),
            j3=int(row['j3']), j4=int(row['j4']),
            hour=int(row['hour']),
            day_of_week=int(row['day_of_week']),
            sim_label=str(row['DateTime']),
            placeholders=placeholders
        )
        progress_bar.progress(min(int((step+1)/total_steps*100), 100))
        time.sleep(sim_speed)

    st.success("✅ Simulation complete!")

else:
    render_dashboard(j1, j2, j3, j4, hour, day_of_week)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption(
    "Smart Traffic Light Digital Twin · "
    "Python · Scikit-learn · Streamlit · "
    "Random Forest · Chronological validation"
)