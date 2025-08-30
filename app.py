from flask import Flask
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import os

# ------------------ ML MODELS ------------------ #
model = joblib.load("decision_tree.pkl")
encoder = joblib.load("encoder.pkl")
selected_features = joblib.load("selected_features.pkl")

# ------------------ FIREBASE SETUP ------------------ #
cred = credentials.Certificate("/etc/secrets/projectIot.json")

firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-sensors-48dda-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

app = Flask(__name__)

def process_sensor_data(equipment_name, temp, vib, pres, last_temp, last_vib, last_pres):
    temp_change = temp - last_temp
    vib_change  = vib - last_vib
    pres_change = pres - last_pres

    temp_change_pct = (temp_change / last_temp) * 100 if last_temp != 0 else 0
    vib_change_pct  = (vib_change / last_vib) * 100 if last_vib != 0 else 0
    pres_change_pct = (pres_change / last_pres) * 100 if last_pres != 0 else 0

    eq_code = encoder.transform([equipment_name])[0]

    feature_row = {
        'equipment_code': eq_code,
        'temperature': temp,
        'vibration': vib,
        'pressure': pres,
        'temp_change': temp_change,
        'vibration_change': vib_change,
        'pressure_change': pres_change,
        'temp_change_pct': temp_change_pct,
        'vibration_change_pct': vib_change_pct,
        'pressure_change_pct': pres_change_pct
    }

    X = np.array([[feature_row[f] for f in selected_features]])
    risk = model.predict_proba(X)[0][1] * 100

    recommended_action = "None"
    status = "normal"

    if risk > 85:
        status = "warning"

        node_indicator = model.decision_path(X)
        feature_index = model.tree_.feature
        node_ids = node_indicator.indices
        last_split = node_ids[-2] if len(node_ids) > 1 else node_ids[0]

        dominant_feature = selected_features[feature_index[last_split]]

        mapping = {
            "temperature": "temperature",
            "temp_change": "temperature",
            "temp_change_pct": "temperature",
            "vibration": "vibration",
            "vibration_change": "vibration",
            "vibration_change_pct": "vibration",
            "pressure": "pressure",
            "pressure_change": "pressure",
            "pressure_change_pct": "pressure",
        }

        sensor_problem = "unknown"
        for key in mapping:
            if key in dominant_feature:
                sensor_problem = mapping[key]
                break

        recommended_action = f"High {sensor_problem}"

    return {
        "risk_score": float(risk),
        "status": status,
        "recommended_action": recommended_action
    }

# ------------------ LISTENER ------------------ #
def listener(event):
    data = event.data
    if not isinstance(data, dict):
        return
    
    equipment = data.get("equipment", "pump101")
    temp = float(data.get("temperture", 0))
    vib = float(data.get("viberation", 0))
    pres = float(data.get("pressure", 0))

    last_data = db.reference(f"Predictions/{equipment}/last").get() or {"temp":0,"vib":0,"pres":0}

    result = process_sensor_data(
        equipment, temp, vib, pres,
        last_data["temp"], last_data["vib"], last_data["pres"]
    )

    db.reference(f"Predictions/{equipment}").set({
        "risk_score": result["risk_score"],
        "status": result["status"],
        "recommended_action": result["recommended_action"],
        "last": {"temp": temp, "vib": vib, "pres": pres}
    })

sensor_ref = db.reference("Sensors")
sensor_ref.listen(listener)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)


