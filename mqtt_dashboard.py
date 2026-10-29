
import json
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import paho.mqtt.client as mqtt
from sklearn.ensemble import IsolationForest


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Industrial Motor Monitoring",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ AI-Based Industrial Motor Monitoring System")
st.caption(
    "Live IoT sensor monitoring using Wokwi, MQTT, Streamlit, "
    "and machine learning."
)


# =========================================================
# CONFIGURATION
# =========================================================

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "motor-monitor-demo-7f3a/sensors"

MAX_READINGS = 300

VIBRATION_LIMIT = 5.0
TEMPERATURE_LIMIT = 70.0
CURRENT_LIMIT = 1.0


# =========================================================
# AI MODEL
# =========================================================

@st.cache_resource
def load_ai_model():
    """Train an Isolation Forest using simulated normal readings."""

    rng = np.random.default_rng(42)

    normal_data = np.column_stack(
        [
            rng.normal(2.0, 0.7, 1000),   # Vibration
            rng.normal(45.0, 8.0, 1000),  # Temperature
            rng.normal(0.55, 0.15, 1000), # Current
        ]
    )

    normal_data = np.clip(
        normal_data,
        a_min=[0.0, 15.0, 0.0],
        a_max=[4.9, 69.0, 0.99],
    )

    model = IsolationForest(
        n_estimators=150,
        contamination=0.05,
        random_state=42,
    )

    model.fit(normal_data)
    return model


model = load_ai_model()


# =========================================================
# SHARED MQTT DATA
# =========================================================

@st.cache_resource
def create_shared_store():
    return {
        "readings": deque(maxlen=MAX_READINGS),
        "lock": threading.Lock(),
        "connected": False,
        "last_message": None,
        "error": None,
    }


store = create_shared_store()


# =========================================================
# MQTT CALLBACKS
# =========================================================

def on_connect(client, userdata, flags, reason_code, properties=None):
    """Subscribe when the MQTT connection succeeds."""

    try:
        success = int(reason_code) == 0
    except (TypeError, ValueError):
        success = str(reason_code).lower() in ("success", "0")

    with store["lock"]:
        store["connected"] = success
        store["error"] = None if success else (
            f"MQTT connection failed: {reason_code}"
        )

    if success:
        client.subscribe(MQTT_TOPIC)
        print(f"Connected to MQTT broker. Subscribed to {MQTT_TOPIC}")
    else:
        print(f"MQTT connection failed: {reason_code}")


def on_disconnect(
    client,
    userdata,
    disconnect_flags=None,
    reason_code=None,
    properties=None,
):
    with store["lock"]:
        store["connected"] = False


def on_message(client, userdata, message):
    """Receive JSON sensor data and save a valid reading."""

    try:
        payload = message.payload.decode("utf-8")
        data = json.loads(payload)

        # Support common key names from the Wokwi ESP32 code.
        vibration = data.get("vibration", data.get("vibration_value"))
        temperature = data.get("temperature", data.get("temp"))
        current = data.get("current", data.get("motor_current"))

        if vibration is None or temperature is None or current is None:
            raise ValueError(
                "JSON must contain vibration, temperature, and current."
            )

        reading = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "vibration": float(vibration),
            "temperature": float(temperature),
            "current": float(current),
        }

        with store["lock"]:
            store["readings"].append(reading)
            store["last_message"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            store["error"] = None

    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
        with store["lock"]:
            store["error"] = f"Invalid sensor message: {exc}"


# =========================================================
# START MQTT CLIENT
# =========================================================

@st.cache_resource
def start_mqtt():
    """Create one MQTT client shared across Streamlit reruns."""

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION1,
        client_id="motor-dashboard-" + str(np.random.randint(100000, 999999)),
        protocol=mqtt.MQTTv311,
    )

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
    except Exception as exc:
        with store["lock"]:
            store["connected"] = False
            store["error"] = str(exc)

    return client


mqtt_client = start_mqtt()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ System Configuration")
st.sidebar.write(f"**Broker:** `{MQTT_BROKER}`")
st.sidebar.write(f"**Port:** `{MQTT_PORT}`")
st.sidebar.write(f"**Topic:** `{MQTT_TOPIC}`")

st.sidebar.divider()
st.sidebar.subheader("Fault Thresholds")
st.sidebar.write(f"Vibration: > {VIBRATION_LIMIT}")
st.sidebar.write(f"Temperature: > {TEMPERATURE_LIMIT} °C")
st.sidebar.write(f"Current: > {CURRENT_LIMIT} A")

st.sidebar.divider()
st.sidebar.info(
    "Keep the Wokwi simulation running and publishing JSON messages "
    "to the configured MQTT topic."
)


# =========================================================
# LIVE DASHBOARD
# =========================================================

@st.fragment(run_every="2s")
def live_dashboard():

    with store["lock"]:
        connected = store["connected"]
        error = store["error"]
        last_message = store["last_message"]
        readings = list(store["readings"])

    # Connection status
    if connected:
        st.success("MQTT broker connected")
    else:
        st.warning(
            "Connecting to MQTT broker. If this persists, check the "
            "broker, network, and Wokwi connection."
        )

    if error:
        st.error(error)

    if last_message:
        st.caption(f"Last valid sensor message: {last_message}")

    if not readings:
        st.info(
            "Waiting for sensor readings. Start Wokwi and ensure it "
            "publishes JSON to the exact topic shown in the sidebar."
        )
        return

    df = pd.DataFrame(readings)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Latest reading
    latest = df.iloc[-1]

    vibration = float(latest["vibration"])
    temperature = float(latest["temperature"])
    current = float(latest["current"])

    # AI anomaly detection
    features = np.array([[vibration, temperature, current]])
    prediction = int(model.predict(features)[0])
    anomaly_score = float(model.decision_function(features)[0])
    ai_anomaly = prediction == -1

    # Threshold checks
    reasons = []

    if vibration > VIBRATION_LIMIT:
        reasons.append("High vibration")

    if temperature > TEMPERATURE_LIMIT:
        reasons.append("High temperature")

    if current > CURRENT_LIMIT:
        reasons.append("High current")

    if ai_anomaly:
        reasons.append("AI detected anomaly")

    motor_fault = bool(reasons)
    motor_status = "FAULT DETECTED" if motor_fault else "NORMAL"

    # -----------------------------------------------------
    # FAULT HISTORY
    # -----------------------------------------------------

    if "fault_history" not in st.session_state:
        st.session_state.fault_history = []

    if "previous_fault" not in st.session_state:
        st.session_state.previous_fault = False

    if motor_fault and not st.session_state.previous_fault:
        st.session_state.fault_history.append(
            {
                "Start Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "Fault Type": ", ".join(reasons),
                "Vibration": vibration,
                "Temperature": temperature,
                "Current": current,
                "Status": "Active",
            }
        )

    elif not motor_fault and st.session_state.previous_fault:
        history = st.session_state.fault_history

        for event in reversed(history):
            if event["Status"] == "Active":
                event["End Time"] = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                event["Status"] = "Cleared"
                break

    st.session_state.previous_fault = motor_fault

    # -----------------------------------------------------
    # MOTOR STATUS
    # -----------------------------------------------------

    st.subheader("🩺 Motor Health Status")

    if motor_fault:
        st.error("⚠️ MOTOR FAULT: " + ", ".join(reasons))
    else:
        st.success("✅ MOTOR OPERATING NORMALLY")

    # -----------------------------------------------------
    # KPI METRICS
    # -----------------------------------------------------

    st.subheader("📊 Live Sensor Readings")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Vibration",
        f"{vibration:.2f}",
        help="Simulated vibration reading",
    )

    col2.metric(
        "Temperature",
        f"{temperature:.2f} °C",
        help="Simulated motor temperature",
    )

    col3.metric(
        "Current",
        f"{current:.2f} A",
        help="Simulated motor current",
    )

    col4.metric(
        "Readings Received",
        len(df),
    )

    # -----------------------------------------------------
    # CHARTS
    # -----------------------------------------------------

    st.subheader("📈 Real-Time Sensor Trends")

    chart1, chart2 = st.columns(2)

    with chart1:
        fig_vibration = px.line(
            df,
            x="timestamp",
            y="vibration",
            title="Vibration Trend",
            markers=True,
        )
        fig_vibration.add_hline(
            y=VIBRATION_LIMIT,
            line_dash="dash",
            annotation_text="Fault threshold",
        )
        st.plotly_chart(
            fig_vibration,
            use_container_width=True,
        )

    with chart2:
        fig_temperature = px.line(
            df,
            x="timestamp",
            y="temperature",
            title="Temperature Trend (°C)",
            markers=True,
        )
        fig_temperature.add_hline(
            y=TEMPERATURE_LIMIT,
            line_dash="dash",
            annotation_text="Fault threshold",
        )
        st.plotly_chart(
            fig_temperature,
            use_container_width=True,
        )

    fig_current = px.line(
        df,
        x="timestamp",
        y="current",
        title="Motor Current Trend (A)",
        markers=True,
    )
    fig_current.add_hline(
        y=CURRENT_LIMIT,
        line_dash="dash",
        annotation_text="Fault threshold",
    )
    st.plotly_chart(
        fig_current,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # AI RESULTS
    # -----------------------------------------------------

    st.subheader("🤖 AI Anomaly Detection")

    ai_col1, ai_col2, ai_col3 = st.columns(3)

    ai_col1.metric(
        "AI Result",
        "Anomaly" if ai_anomaly else "Normal",
    )

    ai_col2.metric(
        "Anomaly Score",
        f"{anomaly_score:.4f}",
    )

    ai_col3.metric(
        "Motor Status",
        motor_status,
    )

    st.caption(
        "The Isolation Forest model is trained on synthetic normal "
        "data for demonstration. Its predictions are not a validated "
        "industrial diagnosis."
    )

    # -----------------------------------------------------
    # FAULT HISTORY TABLE
    # -----------------------------------------------------

    st.subheader("🚨 Fault Event History")

    history = st.session_state.fault_history

    if history:
        history_df = pd.DataFrame(history)
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Fault History CSV",
            data=csv_data,
            file_name="motor_fault_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No fault events recorded yet.")

    # -----------------------------------------------------
    # FAULT ANALYTICS
    # -----------------------------------------------------

    st.subheader("📊 Fault Analytics")

    if history:
        history_df = pd.DataFrame(history)

        # Split combined reasons into individual fault categories.
        fault_types = []

        for event in history:
            for fault_type in event["Fault Type"].split(", "):
                fault_types.append(fault_type)

        if fault_types:
            reason_counts = (
                pd.Series(fault_types)
                .value_counts()
                .rename_axis("Fault Type")
                .reset_index(name="Count")
            )

            fig_faults = px.bar(
                reason_counts,
                x="Fault Type",
                y="Count",
                color="Fault Type",
                title="Recorded Fault Reasons",
                text="Count",
            )

            st.plotly_chart(
                fig_faults,
                use_container_width=True,
            )

        total_events = len(history)
        active_events = sum(
            event["Status"] == "Active" for event in history
        )
        cleared_events = sum(
            event["Status"] == "Cleared" for event in history
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Total Fault Events", total_events)
        k2.metric("Active Fault Events", active_events)
        k3.metric("Cleared Fault Events", cleared_events)

    else:
        st.info(
            "No fault events recorded yet. Keep Wokwi running and "
            "test abnormal sensor values to generate an event."
        )

    # -----------------------------------------------------
    # RAW DATA
    # -----------------------------------------------------

    with st.expander("🔍 View Raw Sensor Data"):
        st.dataframe(
            df.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        sensor_csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Sensor Readings CSV",
            data=sensor_csv,
            file_name="motor_sensor_readings.csv",
            mime="text/csv",
        )


# =========================================================
# RUN DASHBOARD
# =========================================================

live_dashboard()

st.divider()
st.caption(
    "Academic simulation project | Wokwi + ESP32 simulation + MQTT "
    "+ Python Streamlit + Scikit-learn"
)
