# AI-Based Industrial Motor Monitoring System

## Project Overview
A software-based industrial motor monitoring prototype that simulates motor sensor readings, transfers data through MQTT, and displays live monitoring results using a Python Streamlit dashboard.

## Technologies Used
- Python
- Streamlit
- MQTT
- Wokwi ESP32 simulation
- Pandas and NumPy
- Plotly
- Scikit-learn — Isolation Forest

## System Architecture
Wokwi ESP32 Simulation → MQTT Broker → Python Streamlit Dashboard → AI Anomaly Detection → Fault Alerts and History

## Features
- Live vibration, temperature, and current readings
- Real-time sensor charts
- Threshold-based fault alerts
- Prototype AI anomaly detection
- Fault event history and analytics
- CSV export of sensor readings and fault history

## How to Run
1. Open the project folder in VS Code.
2. Activate the Python virtual environment.
3. Install dependencies:
   `pip install streamlit numpy pandas plotly scikit-learn paho-mqtt`
4. Start the Wokwi simulation and publish sensor data to the configured MQTT topic.
5. Run:
   `streamlit run mqtt_dashboard.py`

## Project Limitations
This is a software simulation, not a physical industrial motor installation. The AI model uses synthetic training data and requires real sensor data and validation before any industrial deployment.

## Future Improvements
- Train and validate the model using real motor sensor datasets.
- Add persistent database storage.
- Add sensor calibration and configurable thresholds.
- Test with physical sensors and an ESP32.
