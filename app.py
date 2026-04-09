import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Bengaluru Traffic Predictor", layout="wide")

# Custom CSS for modern styling
st.markdown("""
<style>
/* Base theme */
body, .stApp {
    background-color: #121212;
    color: #f1f1f1;
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1e1e1e !important;
    border-right: 1px solid #333;
}

h1, h2, h3 {
    color: #bb86fc;
}

/* Custom cards for metric and outputs */
div.css-1r6slb0.e1tzin5v2 {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}

/* Enhancing inputs */
.stSelectbox label, .stSlider label {
    font-weight: 600;
    color: #bb86fc;
}

/* Highlighted button */
div.stButton > button {
    background-color: #bb86fc;
    color: #121212;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: bold;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background-color: #3700b3;
    color: #ffffff;
    box-shadow: 0 4px 8px rgba(0,0,0,0.5);
}

/* Container styling */
.glass-container {
    background: rgba(30, 30, 30, 0.7);
    backdrop-filter: blur(10px);
    border-radius: 12px;
    padding: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
</style>
""", unsafe_allow_html=True)

# Load Model Configuration
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        return dict(data)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_data = load_model()

if model_data:
    model = model_data['model']
    encoder = model_data['encoder']
    distance_mapping = model_data['distance_mapping']
    options = model_data['options']
    categorical_cols = model_data['categorical_cols']
    numerical_cols = model_data['numerical_cols']
    features_cols = model_data['features']

    # Title
    st.title("🚗 Bengaluru Traffic Predictor")
    st.markdown("Predict travel times using our Random Forest ML model based on Bengaluru static datasets.")

    # Sidebar Navigation
    st.sidebar.header("Navigation")
    mode = st.sidebar.radio("Select Mode", ["Time Travel", "What-If Analysis", "Comparison"])

    def predict_travel_time(start_area, end_area, time_of_day, day_of_week, weather, density, road):
        # Determine distance
        dist = distance_mapping.get((start_area, end_area))
        if dist is None:
             # fallback to reverse or average
            dist = distance_mapping.get((end_area, start_area), 10.0)

        input_data = pd.DataFrame([{
            'start_area': start_area,
            'end_area': end_area,
            'time_of_day': time_of_day,
            'day_of_week': day_of_week,
            'weather_condition': weather,
            'traffic_density_level': density,
            'road_type': road,
            'distance_km': dist
        }])

        # Encode categorical
        encoded_cat = encoder.transform(input_data[categorical_cols])
        encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

        # Combine
        X = pd.concat([input_data[numerical_cols], encoded_df], axis=1)

        # Ensure order
        X = X[features_cols]
        return model.predict(X)[0], dist

    if mode == "Time Travel":
        st.header("🕰️ Time Travel Mode")
        st.write("Simulate future variables to predict commute duration.")
        
        col1, col2 = st.columns(2)
        with col1:
            start_area = st.selectbox("Start Location", options['start_areas'])
            day_of_week = st.selectbox("Day of the week (e.g. Next Tuesday = Weekday)", options['day_of_week'])
            weather = st.selectbox("Forecasted weather", options['weather_condition'])
            density = st.selectbox("Expected Traffic Density", options['traffic_density_level'], index=1)
            
        with col2:
            end_area = st.selectbox("End Location", options['end_areas'], index=1)
            time_of_day = st.selectbox("Time of Day (e.g. 6 PM = Evening Peak)", options['time_of_day'])
            road = st.selectbox("Route Road Type", options['road_types'])
            
        if st.button("Predict Duration"):
            pred_time, dist = predict_travel_time(start_area, end_area, time_of_day, day_of_week, weather, density, road)
            st.markdown("<div class='glass-container'>", unsafe_allow_html=True)
            col_res1, col_res2 = st.columns(2)
            col_res1.metric(label="Predicted Travel Time", value=f"{pred_time:.1f} mins")
            col_res2.metric(label="Estimated Distance", value=f"{dist:.1f} km")
            st.markdown("</div>", unsafe_allow_html=True)

    elif mode == "What-If Analysis":
        st.header("🤔 What-If Analysis")
        st.write("See how small changes affect your known route.")
        
        c1, c2 = st.columns(2)
        with c1:
            start_area = st.selectbox("Start", options['start_areas'])
            end_area = st.selectbox("End", list(reversed(options['end_areas'])))
        
        st.write("### Base Constraints")
        c3, c4 = st.columns(2)
        with c3:
            time_of_day = st.selectbox("Time of Day", options['time_of_day'])
            road = st.selectbox("Road Type", options['road_types'])
            density = st.selectbox("Density", options['traffic_density_level'])

        st.write("### Toggle Scenarios")
        scenario_type = st.radio("What do you want to toggle?", ["Weather (Clear vs Heavy Rain)", "Day Type (Weekday vs Weekend)"])

        if st.button("Compare Scenarios"):
            if scenario_type.startswith("Weather"):
                time1, _ = predict_travel_time(start_area, end_area, time_of_day, options['day_of_week'][0], "Clear", density, road)
                time2, _ = predict_travel_time(start_area, end_area, time_of_day, options['day_of_week'][0], "Rain", density, road)
                label1, label2 = "Clear Weather", "Heavy Rain"
            else:
                time1, _ = predict_travel_time(start_area, end_area, time_of_day, "Weekday", options['weather_condition'][0], density, road)
                time2, _ = predict_travel_time(start_area, end_area, time_of_day, "Weekend", options['weather_condition'][0], density, road)
                label1, label2 = "Weekday", "Weekend"

            diff = time2 - time1
            
            mc1, mc2 = st.columns(2)
            mc1.metric(label=label1, value=f"{time1:.1f} mins")
            mc2.metric(label=label2, value=f"{time2:.1f} mins", delta=f"{diff:+.1f} mins", delta_color="inverse")

    elif mode == "Comparison":
        st.header("⚖️ Comparison Mode")
        st.write("Pit two entirely different situations against one another.")
        
        col_A, col_B = st.columns(2)
        with col_A:
            st.subheader("Scenario A")
            start_A = st.selectbox("Start A", options['start_areas'], key="sA")
            end_A = st.selectbox("End A", options['end_areas'], key="eA")
            time_A = st.selectbox("Time A", options['time_of_day'], key="tA")
            day_A = st.selectbox("Day A", options['day_of_week'], key="dA")
            weather_A = st.selectbox("Weather A", options['weather_condition'], key="wA")
            
        with col_B:
            st.subheader("Scenario B")
            start_B = st.selectbox("Start B", options['start_areas'], key="sB")
            end_B = st.selectbox("End B", reversed(options['end_areas']), key="eB")
            time_B = st.selectbox("Time B", reversed(options['time_of_day']), key="tB")
            day_B = st.selectbox("Day B", reversed(options['day_of_week']), key="dB")
            weather_B = st.selectbox("Weather B", reversed(options['weather_condition']), key="wB")
            
        if st.button("Run Comparison"):
            time_a, dist_a = predict_travel_time(start_A, end_A, time_A, day_A, weather_A, options['traffic_density_level'][0], options['road_types'][0])
            time_b, dist_b = predict_travel_time(start_B, end_B, time_B, day_B, weather_B, options['traffic_density_level'][0], options['road_types'][0])
            
            res1, res2 = st.columns(2)
            res1.metric("Scenario A Duration", f"{time_a:.1f} mins", f"Dist: {dist_a:.1f} km", delta_color="off")
            res2.metric("Scenario B Duration", f"{time_b:.1f} mins", f"Dist: {dist_b:.1f} km", delta_color="off")
            
            chart_data = pd.DataFrame(
               {"Duration (mins)": [time_a, time_b]},
               index=["Scenario A", "Scenario B"]
            )
            st.bar_chart(chart_data)
else:
    st.warning("model.pkl not found! Please run the training script first.")
