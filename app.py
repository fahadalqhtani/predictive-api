from flask import Flask, request, jsonify
import joblib
import numpy as np

# تحميل الموديل
model = joblib.load("decision_tree.pkl")

# تحميل الـ encoder وقائمة الميزات
encoder = joblib.load("encoder.pkl")
selected_features = joblib.load("selected_features.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        equipment_name = data.get("equipment")
        temp = data.get("temperature")
        vib  = data.get("vibration")
        pres = data.get("pressure")
        last_temp = data.get("last_temp", 0)
        last_vib = data.get("last_vib", 0)
        last_pres = data.get("last_pressure", 0)

        # حساب المشتقات
        temp_change = temp - last_temp
        vib_change  = vib - last_vib
        pres_change = pres - last_pres

        temp_change_pct = (temp_change / last_temp) * 100 if last_temp != 0 else 0
        vib_change_pct  = (vib_change / last_vib) * 100 if last_vib != 0 else 0
        pres_change_pct = (pres_change / last_pres) * 100 if last_pres != 0 else 0

        # ترميز اسم الجهاز
        eq_code = encoder.transform([equipment_name])[0]

        # بناء feature row
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

        # اختيار الميزات مثل التدريب
        X = np.array([[feature_row[f] for f in selected_features]])

        # التنبؤ
        risk = model.predict_proba(X)[0][1] * 100

        # إذا الخطر عالي (>85%) نحدد أي حساس السبب
        recommended_action = "None"
        if risk > 85:
            node_indicator = model.decision_path(X)
            feature_index = model.tree_.feature
            node_ids = node_indicator.indices
            last_split = node_ids[-2] if len(node_ids) > 1 else node_ids[0]

            dominant_feature = selected_features[feature_index[last_split]]

            # ✅ خريطة الميزات المشتقة → المستشعر الأساسي
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

        return jsonify({
            "risk_score": float(risk),
            "status": "warning" if risk > 85 else "normal",
            "recommended_action": recommended_action
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
