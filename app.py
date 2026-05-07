import streamlit as st
import pandas as pd
import pickle
import streamlit.components.v1 as components
import json

st.set_page_config(page_title="Traffic Predictor", layout="wide", initial_sidebar_state="collapsed")

# ── CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Hide sidebar & defaults */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
footer, [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent !important; z-index: 100 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; margin: 0 !important; }

body, .stApp { background-color: #0a1628 !important; color: #e0e6ed; font-family: 'Inter', sans-serif; overflow: hidden; }
h1,h2,h3,h4,h5,h6 { color: #e0e6ed !important; font-family: 'Inter', sans-serif; }

/* Full-bleed Map iframe */
iframe[title="streamlit_components.v1.components.html"] {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 0 !important;
    border: none !important;
}

/* Navbar */
.top-navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    background: linear-gradient(135deg, rgba(15,29,50,0.85) 0%, rgba(22,37,68,0.85) 100%);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.08);
    padding: 12px 28px; display: flex; align-items: center; justify-content: space-between;
    z-index: 50;
}
.nav-brand { display: flex; align-items: center; gap: 12px; }
.nav-brand svg { width: 24px; height: 24px; color: #00f2fe; }
.nav-brand .title { font-size: 1.35rem; font-weight: 700; color: #fff; letter-spacing: -0.3px; }
.nav-brand .subtitle { font-size: 0.85rem; color: #7b8fad; margin-left: 8px; font-weight: 500; }
.nav-actions { display: flex; align-items: center; gap: 12px; }
.nav-btn {
    background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
    color: #c0cfe0; padding: 7px 16px; border-radius: 8px; font-size: 0.82rem;
    cursor: pointer; transition: all 0.2s; font-weight: 500; display: flex; align-items: center; gap: 8px;
}
.nav-btn:hover { background: rgba(79,172,254,0.15); border-color: rgba(79,172,254,0.3); color: #fff; }
.nav-btn.primary { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: #0a1628; font-weight: 600; border: none; }
.nav-btn.primary:hover { box-shadow: 0 4px 15px rgba(79,172,254,0.4); }
.nav-btn svg { width: 14px; height: 14px; }

/* Floating Left Panel (Simulation Controls) */
div[data-testid="stVerticalBlock"]:has(> div.element-container .floating-left-anchor) {
    position: fixed !important;
    top: 80px !important;
    left: 20px !important;
    width: 360px !important;
    z-index: 10 !important;
    background: rgba(17, 29, 53, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
    max-height: calc(100vh - 100px) !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

/* Custom Scrollbar for Left Panel */
div[data-testid="stVerticalBlock"]:has(> div.element-container .floating-left-anchor)::-webkit-scrollbar { width: 6px; }
div[data-testid="stVerticalBlock"]:has(> div.element-container .floating-left-anchor)::-webkit-scrollbar-track { background: transparent; }
div[data-testid="stVerticalBlock"]:has(> div.element-container .floating-left-anchor)::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }

/* Form inputs styling */
.stSelectbox label, .stSlider label, .stNumberInput label { font-weight: 500 !important; color: #9bb0cc !important; font-size: 0.8rem !important; }
.stSelectbox > div > div, .stNumberInput > div > div > input { background-color: rgba(10,22,40,0.6) !important; border-color: rgba(255,255,255,0.08) !important; color: #e0e6ed !important; border-radius: 8px !important; }
div.stButton > button {
    background: linear-gradient(135deg, #146bf7 0%, #00f2fe 100%) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: 12px 24px !important; font-weight: 600 !important; font-size: 0.95rem !important;
    transition: all 0.3s ease !important; width: 100%; margin-top: 10px;
}
div.stButton > button:hover { box-shadow: 0 6px 20px rgba(79,172,254,0.3) !important; transform: translateY(-1px); }

.stExpander { border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 12px !important; background: rgba(10,22,40,0.4) !important; }
.stExpander summary p { font-weight: 500 !important; color: #9bb0cc !important; }

/* Right Panel (Metrics) purely HTML */
.floating-right {
    position: fixed;
    top: 80px;
    right: 20px;
    width: 440px;
    z-index: 10;
    pointer-events: none; /* Let clicks pass through empty areas */
}
.floating-right > div { pointer-events: auto; } /* Re-enable clicks on the actual cards */

.metrics-card {
    background: rgba(17, 29, 53, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.metrics-card h3 {
    font-size: 1rem; font-weight: 600; color: #fff; margin: 0 0 16px 0;
    display: flex; align-items: center; gap: 8px;
}
.metrics-card h3 svg { width: 18px; height: 18px; color: #4facfe; }

/* Grid for Output Stats */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}
.stat-box {
    background: rgba(10, 22, 40, 0.5);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 16px;
    display: flex; flex-direction: column; align-items: center; text-align: center;
}
.stat-box svg { width: 24px; height: 24px; color: #4facfe; margin-bottom: 8px; }
.stat-box.highlight svg { color: #14b8a6; }
.stat-label { font-size: 0.75rem; color: #7b8fad; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; font-weight: 600; }
.stat-value { font-size: 1.4rem; font-weight: 700; color: #fff; }
.stat-unit { font-size: 0.8rem; font-weight: 500; color: #9bb0cc; margin-left: 2px; }
.stat-sub { font-size: 0.7rem; color: #14b8a6; margin-top: 4px; background: rgba(20,184,166,0.1); padding: 2px 6px; border-radius: 4px; }

/* Route Breakdown / Legend */
.route-breakdown { font-size: 0.85rem; }
.route-item { display: flex; align-items: center; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
.route-item:last-child { border-bottom: none; padding-bottom: 0; }
.route-dot { width: 8px; height: 8px; border-radius: 50%; margin-right: 12px; flex-shrink: 0; }
.route-name { color: #c0cfe0; flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; padding-right: 10px; }
.route-vol { color: #fff; font-weight: 600; font-variant-numeric: tabular-nums; }

/* Download floating button */
.dl-floating {
    position: fixed; bottom: 24px; left: 24px; z-index: 50;
    background: rgba(17, 29, 53, 0.75); backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 12px;
    padding: 12px 16px; display: flex; align-items: center; gap: 12px;
    cursor: pointer; transition: all 0.2s; color: #fff; text-decoration: none;
}
.dl-floating:hover { background: rgba(20, 34, 60, 0.9); border-color: rgba(255,255,255,0.2); }
.dl-floating svg { width: 18px; height: 18px; color: #4facfe; }
.dl-info { display: flex; flex-direction: column; }
.dl-title { font-size: 0.85rem; font-weight: 600; }
.dl-sub { font-size: 0.7rem; color: #7b8fad; }

</style>
""", unsafe_allow_html=True)

# ── NAVBAR ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-navbar">
    <div class="nav-brand">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="M12 18v4"/><path d="M4.93 4.93l2.83 2.83"/><path d="M16.24 16.24l2.83 2.83"/><path d="M2 12h4"/><path d="M18 12h4"/><path d="M4.93 19.07l2.83-2.83"/><path d="M16.24 7.76l2.83-2.83"/><circle cx="12" cy="12" r="4"/></svg>
        <span class="title">Traffic Predictor</span>
        <span class="subtitle">Bengaluru Live Traffic</span>
    </div>
    <div class="nav-actions">
        <button class="nav-btn" onclick="document.querySelector('[data-testid=\\'stDownloadButton\\'] button').click()">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Download CSV
        </button>
        <button class="nav-btn primary" onclick="window.open('https://streamlit.io/cloud','_blank')">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-6 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4c1.62-1.08 5-4 5-4v5"/><circle cx="15" cy="9" r="1"/></svg>
            Deploy
        </button>
        <button class="nav-btn">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="1"/><circle cx="12" cy="5" r="1"/><circle cx="12" cy="19" r="1"/></svg>
        </button>
    </div>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA ────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    try:
        with open('model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        with open('distances.pkl', 'rb') as f:
            distances = pickle.load(f)
        df = pd.read_csv('Banglore_traffic_Dataset.csv')
        valid_routes = df.groupby('Area Name')['Road/Intersection Name'].unique().apply(list).to_dict()
        return model_data, distances, valid_routes
    except Exception as e:
        st.error(f"Error loading model/distances: {e}")
        return None, None, None

model_data, distances, valid_routes = load_data()

if not model_data:
    st.warning("model.pkl or distances.pkl not found! Please run the training script first.")
    st.stop()

model_vol = model_data['model_vol']
model_spd = model_data['model_spd']
encoder = model_data['encoder']
options = model_data['options']
numerical_limits = model_data['numerical_limits']
categorical_cols = model_data['categorical_cols']
numerical_cols = model_data['numerical_cols']
features_cols = model_data['features']

def predict_metrics(inputs):
    input_data = pd.DataFrame([inputs])
    encoded_cat = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))
    X = pd.concat([input_data[numerical_cols].reset_index(drop=True), encoded_df], axis=1)
    X = X[features_cols]
    vol = model_vol.predict(X)[0]
    spd = model_spd.predict(X)[0]
    return vol, spd

if 'predicted' not in st.session_state:
    st.session_state.predicted = False
    st.session_state.results = {}

# ── FULL BLEED MAP ───────────────────────────────────────────────────────
try:
    with open('bangalore_traffic_map.html', 'r', encoding='utf-8') as f:
        html_data = f.read()
    
    # Inject CSS to move the zoom control to bottom middle
    html_data = html_data.replace('</head>', """
    <style>
    .leaflet-control-zoom {
        position: fixed !important;
        bottom: 20px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        top: auto !important;
    }
    </style>
    </head>
    """)
    # The iframe styling in CSS handles full bleed
    components.html(html_data, height=1000, scrolling=False)
except:
    st.info("Map file not generated yet. Run generate_map.py first.")

# ── LEFT FLOATING PANEL (SIMULATION CONTROLS) ────────────────────────────
with st.container():
    st.markdown('<span class="floating-left-anchor"></span>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fbbd23" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="10" r="3"/><path d="M12 21.7C17.3 17 20 13 20 10a8 8 0 1 0-16 0c0 3 2.7 7 8 11.7z"/></svg>
        <h3 style="margin:0; font-size:1.05rem; font-weight:700;">Simulation Controls</h3>
    </div>
    """, unsafe_allow_html=True)

    inputs = {}
    def round_10(val): return int(round(val / 10.0) * 10)

    inputs['Area Name'] = st.selectbox("Area Name (Start Area)", options['Area Name'])
    available_roads = valid_routes.get(inputs['Area Name'], options['Road/Intersection Name']) if valid_routes else options['Road/Intersection Name']
    inputs['Road/Intersection Name'] = st.selectbox("Road/Intersection (Destination)", available_roads)
    inputs['Weather Conditions'] = st.selectbox("Weather Conditions", options['Weather Conditions'])
    inputs['Roadwork and Construction Activity'] = st.selectbox("Roadwork Activity", options['Roadwork and Construction Activity'])
    
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    inputs['Congestion Level'] = st.slider("Congestion Level (%)",
        min_value=round_10(numerical_limits['Congestion Level']['min']),
        max_value=max(round_10(numerical_limits['Congestion Level']['max']), round_10(numerical_limits['Congestion Level']['min']) + 10),
        value=round_10(numerical_limits['Congestion Level']['mean']), step=10)
    
    inputs['Road Capacity Utilization'] = st.slider("Road Capacity Utilization (%)",
        min_value=round_10(numerical_limits['Road Capacity Utilization']['min']),
        max_value=max(round_10(numerical_limits['Road Capacity Utilization']['max']), round_10(numerical_limits['Road Capacity Utilization']['min']) + 10),
        value=round_10(numerical_limits['Road Capacity Utilization']['mean']), step=10)

    c1, c2 = st.columns(2)
    with c1: inputs['Travel Time Index'] = st.number_input("Travel Time Index", value=numerical_limits['Travel Time Index']['mean'])
    with c2: inputs['Incident Reports'] = st.number_input("Incident Reports", value=numerical_limits['Incident Reports']['mean'])

    with st.expander("Fine-Grained Adjustments"):
        inputs['Public Transport Usage'] = st.number_input("Public Transport Usage", value=numerical_limits['Public Transport Usage']['mean'])
        inputs['Parking Usage'] = st.number_input("Parking Usage", value=numerical_limits['Parking Usage']['mean'])
        inputs['Traffic Signal Compliance'] = st.number_input("Traffic Signal Compliance", value=numerical_limits['Traffic Signal Compliance']['mean'])
        inputs['Pedestrian and Cyclist Count'] = st.number_input("Pedestrian/Cyclist Count", value=numerical_limits['Pedestrian and Cyclist Count']['mean'])

    if st.button("Predict"):
        vol, spd = predict_metrics(inputs)
        route_key = f"{inputs['Area Name']}_{inputs['Road/Intersection Name']}"
        dist_km = distances.get(route_key, 5.0)
        time_mins = (dist_km / max(spd, 1.0)) * 60
        st.session_state.predicted = True
        st.session_state.results = {'vol': vol, 'spd': spd, 'dist': dist_km, 'time': time_mins, 'area': inputs['Area Name'], 'road': inputs['Road/Intersection Name']}
        st.rerun()

# ── RIGHT FLOATING PANEL (METRICS & LEGEND) ──────────────────────────────
right_html = '<div class="floating-right">'

if st.session_state.predicted:
    r = st.session_state.results
    
    if r['vol'] < 20000:
        v_color = "#14b8a6"
    elif r['vol'] < 40000:
        v_color = "#fbbd23"
    else:
        v_color = "#ef4444"

    right_html += f"""
    <div class="metrics-card">
        <h3>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
            Traffic Metrics Overview
        </h3>
        <div class="stats-grid">
            <div class="stat-box highlight">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                <div class="stat-label">Avg. Delivery Time</div>
                <div><span class="stat-value">{r['time']:.0f}</span><span class="stat-unit">min</span></div>
                <div class="stat-sub">Estimated</div>
            </div>
            <div class="stat-box">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/></svg>
                <div class="stat-label">Traffic Volume</div>
                <div><span class="stat-value" style="color:{v_color};">{r['vol']:,.0f}</span><span class="stat-unit">veh</span></div>
            </div>
            <div class="stat-box">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>
                <div class="stat-label">Average Speed</div>
                <div><span class="stat-value">{r['spd']:.1f}</span><span class="stat-unit">km/h</span></div>
            </div>
            <div class="stat-box">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6l6-3 6 3 6-3v12l-6 3-6-3-6 3V6z"/><path d="M9 3v12"/><path d="M15 6v12"/></svg>
                <div class="stat-label">Total Distance</div>
                <div><span class="stat-value">{r['dist']:.1f}</span><span class="stat-unit">km</span></div>
            </div>
        </div>
    </div>"""

# Always show Route Breakdown / Legend
right_html += """
    <div class="metrics-card">
        <h3>
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>
            Traffic Volume Legend
        </h3>
        <div class="route-breakdown">
            <div class="route-item">
                <div class="route-dot" style="background:#14b8a6; box-shadow:0 0 8px rgba(20,184,166,0.6);"></div>
                <div class="route-name">Low Traffic Density</div>
                <div class="route-vol">&lt; 20,000 veh</div>
            </div>
            <div class="route-item">
                <div class="route-dot" style="background:#fbbd23; box-shadow:0 0 8px rgba(251,189,35,0.6);"></div>
                <div class="route-name">Medium Traffic Density</div>
                <div class="route-vol">20k - 40k veh</div>
            </div>
            <div class="route-item">
                <div class="route-dot" style="background:#ef4444; box-shadow:0 0 8px rgba(239,68,68,0.6);"></div>
                <div class="route-name">High Traffic Density</div>
                <div class="route-vol">&gt; 40,000 veh</div>
            </div>
        </div>
    </div>
</div>"""

# Crucial fix: strip newlines from HTML so Streamlit doesn't parse it as Markdown paragraphs!
st.markdown(right_html.replace('\\n', ''), unsafe_allow_html=True)

# ── DOWNLOAD BUTTON (HIDDEN NATIVE) ────────────────────────────────────────
with open('Banglore_traffic_Dataset.csv', 'rb') as f:
    csv_data = f.read()
st.download_button(
    label="Download Dataset",
    data=csv_data,
    file_name='Banglore_traffic_Dataset.csv',
    mime='text/csv',
    key="dl_hidden", # Hidden via CSS, triggered by top-nav button
)

st.markdown("""
<style>
/* Hide the native download button */
[data-testid="stDownloadButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

