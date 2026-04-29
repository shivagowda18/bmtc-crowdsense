"""
Bus Overcrowding Prediction System
Redesigned App — BMTC Orange Theme
Reva University Mini Project | ISE Dept.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="BMTC CrowdSense",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Noto+Sans:wght@400;500;600&display=swap');
:root {
    --bmtc-orange: #E8521A; --bmtc-dark: #1A1A2E;
    --bmtc-card: #16213E;   --bmtc-accent: #F5A623;
}
html, body, [class*="css"] { font-family: 'Noto Sans', sans-serif; background-color: var(--bmtc-dark) !important; color: #EAEAEA !important; }
.stApp { background-color: var(--bmtc-dark) !important; }
.bmtc-header { background: linear-gradient(135deg, #E8521A 0%, #C0390A 50%, #1A1A2E 100%); padding: 2rem 2.5rem 1.5rem; border-radius: 0 0 24px 24px; margin: -1rem -1rem 2rem -1rem; box-shadow: 0 8px 32px rgba(232,82,26,0.3); }
.bmtc-logo { font-family: 'Rajdhani', sans-serif; font-size: 3rem; font-weight: 700; color: white; line-height: 1; }
.bmtc-subtitle { font-size: 0.85rem; color: rgba(255,255,255,0.75); letter-spacing: 2px; text-transform: uppercase; margin-top: 4px; }
.bmtc-tagline { font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; color: var(--bmtc-accent); font-weight: 600; margin-top: 2px; }
.card { background: var(--bmtc-card); border: 1px solid rgba(232,82,26,0.2); border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
.card-title { font-family: 'Rajdhani', sans-serif; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; color: var(--bmtc-orange); margin-bottom: 0.75rem; font-weight: 600; }
.route-badge { display: inline-block; background: var(--bmtc-orange); color: white; font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; font-weight: 700; padding: 4px 14px; border-radius: 8px; }
.result-low { background: linear-gradient(135deg, rgba(39,174,96,0.15), rgba(39,174,96,0.05)); border: 1px solid rgba(39,174,96,0.4); border-left: 5px solid #27AE60; border-radius: 16px; padding: 1.5rem 2rem; }
.result-medium { background: linear-gradient(135deg, rgba(243,156,18,0.15), rgba(243,156,18,0.05)); border: 1px solid rgba(243,156,18,0.4); border-left: 5px solid #F39C12; border-radius: 16px; padding: 1.5rem 2rem; }
.result-high { background: linear-gradient(135deg, rgba(231,76,60,0.15), rgba(231,76,60,0.05)); border: 1px solid rgba(231,76,60,0.4); border-left: 5px solid #E74C3C; border-radius: 16px; padding: 1.5rem 2rem; }
.stat-strip { display: flex; gap: 1rem; margin: 1rem 0; }
.stat-box { flex: 1; background: rgba(232,82,26,0.1); border: 1px solid rgba(232,82,26,0.2); border-radius: 12px; padding: 1rem; text-align: center; }
.stat-val { font-family: 'Rajdhani', sans-serif; font-size: 1.8rem; font-weight: 700; color: var(--bmtc-orange); }
.stat-lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.6; }
.conf-bar-wrap { margin: 8px 0; }
.conf-label { font-size: 0.82rem; display: flex; justify-content: space-between; margin-bottom: 3px; }
.conf-track { background: rgba(255,255,255,0.07); border-radius: 6px; height: 10px; overflow: hidden; }
.conf-fill-low    { background: linear-gradient(90deg, #27AE60, #2ECC71); height: 100%; border-radius: 6px; }
.conf-fill-medium { background: linear-gradient(90deg, #E67E22, #F39C12); height: 100%; border-radius: 6px; }
.conf-fill-high   { background: linear-gradient(90deg, #C0392B, #E74C3C); height: 100%; border-radius: 6px; }
.orange-divider { height: 2px; background: linear-gradient(90deg, var(--bmtc-orange), transparent); margin: 1.5rem 0; border-radius: 2px; }
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { color: rgba(255,255,255,0.7) !important; font-size: 0.8rem !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
div[data-testid="stSelectbox"] > div > div { background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(232,82,26,0.3) !important; border-radius: 10px !important; color: white !important; }
.stButton > button { background: linear-gradient(135deg, #E8521A, #C0390A) !important; color: white !important; border: none !important; border-radius: 12px !important; font-family: 'Rajdhani', sans-serif !important; font-size: 1.1rem !important; font-weight: 700 !important; letter-spacing: 2px !important; padding: 0.6rem 2rem !important; text-transform: uppercase !important; box-shadow: 0 4px 20px rgba(232,82,26,0.4) !important; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "model_artifacts"

ROUTE_STOPS = {
    "500D": {"name": "Hebbal → Silk Board", "stops": ["Hebbala (Canara Bank)", "Manyata Tech Park", "Veerannapalya", "Kalyan Nagar", "C V Raman Nagar", "Tin Factory", "K. R. Puram Rlwy Station", "Marathahalli Bridge (ORR)", "Kadubeesanahalli", "Bellandur", "Iblur", "Central Silk Board (BTM)", "Central Silk Board (Hosur RD)"]},
    "335E": {"name": "Majestic → Whitefield", "stops": ["Majestic (KBS)", "Corporation", "Domlur", "Indiranagar 100 FT RD (6th Mai)", "Marathahalli (Jn. Vartur and O)", "Kundalahalli", "ITPL", "BSNL Office (Whitefield RD)"]},
    "401":  {"name": "KBS → Yelahanka", "stops": ["Majestic (KBS)", "Mehkri Circle", "Hebbala (Canara Bank)", "Kogilu Cross", "Yelahanka Old Town", "Yelahanka"]},
    "250D": {"name": "Banashankari → Rajajinagar", "stops": ["Banashankari Bus Station", "Jayanagar Water Tank", "South End Circle", "Lalbagh West Gate", "K. R. Market", "City Market", "Rajajinagar/Navaranga"]},
    "G5":   {"name": "KBS → Sarjapur", "stops": ["Majestic (KBS)", "K. R. Market", "Lalbagh West Gate", "Koramangala 1st Block", "Koramangala (SONY World)", "Ejipura", "Kaggadasapura Jn (Big Bazaar)", "Sarjapur Road", "Sarjapur"]},
    "201R": {"name": "Hebbal → Electronic City", "stops": ["Hebbala (Canara Bank)", "Manyata Tech Park", "K. R. Puram Rlwy Station", "Marathahalli Bridge (ORR)", "Central Silk Board (ORR)", "Electronic City (Hosur Rd)", "Electronic City"]},
    "356F": {"name": "Yeshwanthpur → BTM Layout", "stops": ["R M C (Yeshwanthpura)", "Rajajinagar/Navaranga", "Bhashyam Circle (Rajajinagar)", "Majestic (KBS)", "South End Circle", "BTM Check Post", "Kuvempu Nagar (BTM Layout)", "HSR Layout"]},
    "600K": {"name": "Koramangala → Indiranagar", "stops": ["Koramangala (80 FT RD)", "Koramangala (SONY World)", "Ejipura", "Domlur", "Indiranagar 100 FT RD (6th Mai)", "Indiranagar 100 FT RD (13th Ma)", "Binnamangala"]},
    "V1":   {"name": "Banashankari → Marathahalli", "stops": ["Banashankari Bus Station", "Jayanagar Water Tank", "Central Silk Board (BTM)", "HSR Layout (BDA Complex)", "HSR Layout", "Bellandur", "Marathahalli (Jn. Vartur and O)", "Marathahalli Bridge (ORR)"]},
    "C9":   {"name": "Majestic → HSR Layout", "stops": ["Majestic (KBS)", "Lalbagh West Gate", "K. R. Market", "Koramangala 1st Block", "Koramangala (80 FT RD)", "Central Silk Board (BTM)", "HSR Layout (BDA Complex)", "HSR (Depot-25)"]},
}

STOP_POPULARITY = {
    "Majestic (KBS)": 3.0, "Central Silk Board (Hosur RD)": 2.8, "Central Silk Board (BTM)": 2.8,
    "Central Silk Board (ORR)": 2.7, "Hebbala (Canara Bank)": 2.5, "Marathahalli Bridge (ORR)": 2.3,
    "K. R. Puram Rlwy Station": 2.2, "Manyata Tech Park": 2.0, "ITPL": 2.0,
    "Electronic City (Hosur Rd)": 1.9, "Electronic City": 1.9, "Indiranagar 100 FT RD (6th Mai)": 1.8,
    "Banashankari Bus Station": 1.7, "K. R. Market": 1.6, "Koramangala (SONY World)": 1.5,
    "Koramangala 1st Block": 1.5, "HSR Layout": 1.4, "BTM Check Post": 1.3, "Bellandur": 1.3,
    "Kadubeesanahalli": 1.2, "Rajajinagar/Navaranga": 1.2, "R M C (Yeshwanthpura)": 1.2,
    "Kundalahalli": 1.1, "Domlur": 1.1, "Jayanagar Water Tank": 1.0, "South End Circle": 1.0,
}

DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

@st.cache_resource
def load_artifacts():
    model    = pickle.load(open(f"{MODEL_DIR}/bus_model.pkl",  "rb"))
    le_route = pickle.load(open(f"{MODEL_DIR}/le_route.pkl",   "rb"))
    le_stop  = pickle.load(open(f"{MODEL_DIR}/le_stop.pkl",    "rb"))
    le_label = pickle.load(open(f"{MODEL_DIR}/le_label.pkl",   "rb"))
    scaler   = pickle.load(open(f"{MODEL_DIR}/scaler.pkl",     "rb"))
    features = pickle.load(open(f"{MODEL_DIR}/features.pkl",   "rb"))
    return model, le_route, le_stop, le_label, scaler, features

model, le_route, le_stop, le_label, scaler, features = load_artifacts()

def predict_occupancy(route_id, stop_name, hour, day_of_week):
    is_weekend = int(day_of_week >= 5)
    stops = ROUTE_STOPS[route_id]["stops"]
    stop_index = stops.index(stop_name) if stop_name in stops else 0
    total_stops = len(stops)
    popularity = STOP_POPULARITY.get(stop_name, 0.7)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos  = np.cos(2 * np.pi * day_of_week / 7)
    route_enc = le_route.transform([route_id])[0] if route_id in le_route.classes_ else 0
    stop_enc  = le_stop.transform([stop_name])[0] if stop_name in le_stop.classes_ else 0
    scale_vals = scaler.transform([[hour, day_of_week, stop_index, total_stops, popularity]])[0]
    row = {"route_id_enc": route_enc, "stop_name_enc": stop_enc,
           "stop_index": scale_vals[2], "total_stops": scale_vals[3],
           "hour": scale_vals[0], "day_of_week": scale_vals[1],
           "is_weekend": is_weekend, "hour_sin": hour_sin, "hour_cos": hour_cos,
           "dow_sin": dow_sin, "dow_cos": dow_cos, "stop_popularity": scale_vals[4]}
    X = pd.DataFrame([row])[features]
    pred_enc   = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    pred_label = le_label.inverse_transform([pred_enc])[0]
    return pred_label, pred_proba

def hourly_forecast(route_id, stop_name, day_of_week):
    hours, labels = [], []
    for h in range(5, 23):
        label, _ = predict_occupancy(route_id, stop_name, h, day_of_week)
        hours.append(h)
        labels.append(label)
    return hours, labels

# HEADER
st.markdown("""
<div class="bmtc-header">
    <div style="display:flex;align-items:center;gap:1.5rem">
        <div style="font-size:3.5rem">🚌</div>
        <div>
            <div class="bmtc-logo">BMTC CrowdSense</div>
            <div class="bmtc-subtitle">Bengaluru Metropolitan Transport Corporation</div>
            <div class="bmtc-tagline">Real-time Bus Occupancy Prediction System</div>
        </div>
        <div style="margin-left:auto;text-align:right;opacity:0.6;font-size:0.75rem">
            Reva University<br>Mini Project 2024<br>ISE Department
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# STATS
st.markdown("""
<div class="stat-strip">
    <div class="stat-box"><div class="stat-val">10</div><div class="stat-lbl">Active Routes</div></div>
    <div class="stat-box"><div class="stat-val">80%</div><div class="stat-lbl">Model Accuracy</div></div>
    <div class="stat-box"><div class="stat-val">60</div><div class="stat-lbl">Bus Capacity</div></div>
    <div class="stat-box"><div class="stat-val">RF</div><div class="stat-lbl">Algorithm</div></div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="card"><div class="card-title">🚏 Route Selection</div>', unsafe_allow_html=True)
    route_id = st.selectbox("Select Route", options=list(ROUTE_STOPS.keys()),
                            format_func=lambda x: f"{x}  —  {ROUTE_STOPS[x]['name']}")
    stop_name = st.selectbox("Select Boarding Stop", options=ROUTE_STOPS[route_id]["stops"])
    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.75rem;background:rgba(232,82,26,0.1);
                border-radius:10px;border:1px solid rgba(232,82,26,0.2)">
        <span class="route-badge">{route_id}</span>
        <span style="margin-left:10px;font-size:0.85rem;opacity:0.7">
            {ROUTE_STOPS[route_id]['name']} · {len(ROUTE_STOPS[route_id]['stops'])} stops
        </span>
    </div></div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card"><div class="card-title">🕐 Travel Time</div>', unsafe_allow_html=True)
    day_of_week = st.selectbox("Day of Week", options=range(7),
                               format_func=lambda d: f"{'🏢' if d < 5 else '🏖️'} {DAY_NAMES[d]}")
    hour = st.slider("Hour of Travel", min_value=5, max_value=22, value=8, format="%d:00")
    if day_of_week < 5:
        if 8 <= hour <= 10:    peak = "🔴 Morning Peak Hour"
        elif 17 <= hour <= 19: peak = "🔴 Evening Peak Hour"
        elif 12 <= hour <= 14: peak = "🟡 Lunch Hour"
        else:                  peak = "🟢 Off-Peak"
    else:
        peak = "🟡 Weekend"
    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.75rem;background:rgba(232,82,26,0.1);
                border-radius:10px;border:1px solid rgba(232,82,26,0.2);font-size:0.9rem">
        {peak} · {DAY_NAMES[day_of_week]} at {hour:02d}:00
    </div></div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_btn = st.columns([1, 2, 1])
with col_btn[1]:
    predict_clicked = st.button("🔍  PREDICT CROWD LEVEL", use_container_width=True)

if predict_clicked:
    label, proba = predict_occupancy(route_id, stop_name, hour, day_of_week)
    STATUS = {
        "Low":    ("result-low",    "🟢", "#27AE60", "COMFORTABLE", "Plenty of space — easy boarding"),
        "Medium": ("result-medium", "🟡", "#F39C12", "MODERATE",    "Some standing — board quickly"),
        "High":   ("result-high",   "🔴", "#E74C3C", "OVERCROWDED", "Very limited space — wait for next bus"),
    }
    css_class, icon, color, status_word, desc = STATUS[label]
    st.markdown("<div class='orange-divider'></div>", unsafe_allow_html=True)
    col_r1, col_r2 = st.columns([3, 2], gap="large")

    with col_r1:
        st.markdown(f"""
        <div class="{css_class}">
            <div style="font-size:0.75rem;letter-spacing:2px;text-transform:uppercase;color:{color};margin-bottom:6px;font-weight:600">Prediction Result</div>
            <div style="font-family:'Rajdhani',sans-serif;font-size:2rem;font-weight:700;color:{color}">{icon} {status_word}</div>
            <div style="font-size:0.95rem;opacity:0.8;margin:4px 0 0">{desc}</div>
            <div style="font-size:0.78rem;opacity:0.55;margin-top:8px">Route {route_id} · {stop_name} · {DAY_NAMES[day_of_week]} at {hour:02d}:00</div>
        </div>
        """, unsafe_allow_html=True)

        classes = le_label.classes_
        color_map = {"Low":("conf-fill-low","#27AE60"),"Medium":("conf-fill-medium","#F39C12"),"High":("conf-fill-high","#E74C3C")}
        conf_html = '<div class="card" style="margin-top:1rem"><div class="card-title">📊 Confidence Breakdown</div>'
        for i, cls in enumerate(classes):
            pct = round(float(proba[i]) * 100, 1)
            fill_cls, clr = color_map[cls]
            conf_html += f'<div class="conf-bar-wrap"><div class="conf-label"><span style="color:{clr}">{cls}</span><span style="font-weight:600">{pct}%</span></div><div class="conf-track"><div class="{fill_cls}" style="width:{pct}%"></div></div></div>'
        conf_html += '</div>'
        st.markdown(conf_html, unsafe_allow_html=True)

    with col_r2:
        hours, hlabels = hourly_forecast(route_id, stop_name, day_of_week)
        color_map2 = {"Low":"#27AE60","Medium":"#E67E22","High":"#E74C3C"}
        y_vals = [{"Low":1,"Medium":2,"High":3}[l] for l in hlabels]
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor("#16213E")
        ax.set_facecolor("#16213E")
        bars = ax.bar(hours, y_vals, color=[color_map2[l] for l in hlabels], width=0.65, edgecolor="none", zorder=3)
        if hour in hours:
            bars[hours.index(hour)].set_edgecolor("#F5A623")
            bars[hours.index(hour)].set_linewidth(2.5)
        ax.set_xticks(hours)
        ax.set_xticklabels([f"{h}" for h in hours], rotation=45, fontsize=7, color="#AAAAAA")
        ax.set_yticks([1,2,3])
        ax.set_yticklabels(["Low","Med","High"], fontsize=8, color="#AAAAAA")
        ax.set_xlabel("Hour of Day", color="#AAAAAA", fontsize=8)
        ax.set_title(f"Hourly Forecast — {DAY_NAMES[day_of_week]}", color="white", fontsize=10, pad=10, fontweight="bold")
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.set_ylim(0, 3.5)
        ax.yaxis.grid(True, color="rgba(255,255,255,0.05)", zorder=0)
        legend_patches = [mpatches.Patch(color="#27AE60",label="Low"), mpatches.Patch(color="#E67E22",label="Medium"), mpatches.Patch(color="#E74C3C",label="High")]
        ax.legend(handles=legend_patches, fontsize=7, facecolor="#1A1A2E", edgecolor="none", labelcolor="white", loc="upper right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.markdown("""
<div style="text-align:center;padding:1rem;opacity:0.35;font-size:0.75rem;border-top:1px solid rgba(255,255,255,0.06);margin-top:1rem">
    BMTC CrowdSense · Random Forest Classifier · Synthetic BMTC Data · Reva University Mini Project 2024
</div>
""", unsafe_allow_html=True)