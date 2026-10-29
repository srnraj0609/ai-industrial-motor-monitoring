
import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="AI Motor Monitor",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ AI Industrial Motor Monitoring")
st.caption("ESP32 project prototype | Synthetic training and test data")

# Train an anomaly detector on simulated normal motor readings.
rng = np.random.default_rng(42)

normal_training_data = np.column_stack([
    rng.uniform(1, 3, 2000),       # vibration
    rng.uniform(35, 50, 2000),     # temperature
    rng.uniform(0.4, 0.8, 2000),   # current
])

model = make_pipeline(
    StandardScaler(),
    IsolationForest(
        contamination=0.03,
        random_state=42,
    ),
)

model.fit(normal_training_data)

st.sidebar.header("Motor Simulation")
mode = st.sidebar.selectbox(
    "Choose motor condition",
    ["Normal", "High vibration", "Overheating", "Multiple faults"],
)

if "readings" not in st.session_state:
    st.session_state.readings = []

if st.sidebar.button("Generate sensor reading"):
    vibration = random.uniform(1, 3)
    temperature = random.uniform(35, 50)
    current = random.uniform(0.4, 0.8)

    if mode == "High vibration":
        vibration = random.uniform(6, 10)
    elif mode == "Overheating":
        temperature = random.uniform(75, 95)
    elif mode == "Multiple faults":
        vibration = random.uniform(6, 10)
        temperature = random.uniform(75, 95)
        current = random.uniform(1.2, 2.0)

    sample = [[vibration, temperature, current]]
    prediction = model.predict(sample)[0]
    ai_status = "ANOMALY" if prediction == -1 else "NORMAL"

    st.session_state.readings.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Vibration": round(vibration, 2),
        "Temperature": round(temperature, 1),
        "Current": round(current, 2),
        "AI Detection": ai_status,
    })

    st.session_state.readings = st.session_state.readings[-100:]

if st.session_state.readings:
    df = pd.DataFrame(st.session_state.readings)
    latest = df.iloc[-1]

    a, b, c, d = st.columns(4)
    a.metric("Vibration", f"{latest['Vibration']} units")
    b.metric("Temperature", f"{latest['Temperature']} °C")
    c.metric("Current", f"{latest['Current']} A")
    d.metric("AI Detection", latest["AI Detection"])

    if latest["AI Detection"] == "ANOMALY":
        st.error("AI detected an unusual sensor pattern. Investigate the simulated motor.")
    else:
        st.success("AI classified this reading as normal.")

    st.subheader("Sensor Trends")
    chart = px.line(
        df,
        x="Time",
        y=["Vibration", "Temperature", "Current"],
        markers=True,
        title="Motor sensor readings",
    )
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Reading History")
    st.dataframe(df.iloc[::-1], use_container_width=True)

    st.subheader("AI Detection Summary")
    counts = df["AI Detection"].value_counts().reset_index()
    counts.columns = ["Detection", "Readings"]
    st.bar_chart(counts, x="Detection", y="Readings")

else:
    st.info("Choose a motor condition and generate a sensor reading.")

st.caption(
    "Educational prototype: the model is trained on synthetic normal data. "
    "Its predictions are not validated for real industrial motors."
)
