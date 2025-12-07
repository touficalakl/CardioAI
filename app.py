from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ==============================
#  LOAD TRAINED MODEL + SCALER
# ==============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# The model was trained on these columns in this EXACT order:
# ["HeartRate", "O2Saturation", "ECG", "Echo", "Troponin",
#  "ECGIschemicChanges", "ChestPainType", "BNP",
#  "Hypertension", "STChanges", "WallMotionAbnormalities"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    prob = None

    # default values for the form (so it remembers last input)
    values = {
        "HeartRate": "",
        "O2Saturation": "",
        "ECG": "0",
        "Echo": "0",
        "Troponin": "",
        "ECGIschemicChanges": "0",
        "ChestPainType": "0",
        "BNP": "",
        "Hypertension": "0",
        "STChanges": "0",
        "WallMotionAbnormalities": "0",
    }

    if request.method == "POST":
        # 1. read values from the HTML form
        for key in values.keys():
            values[key] = request.form.get(key, "")

        try:
            # 2. build feature vector in the SAME order as training
            features = np.array([[
                float(values["HeartRate"]),
                float(values["O2Saturation"]),
                int(values["ECG"]),
                int(values["Echo"]),
                float(values["Troponin"]),
                int(values["ECGIschemicChanges"]),
                int(values["ChestPainType"]),
                float(values["BNP"]),
                int(values["Hypertension"]),
                int(values["STChanges"]),
                int(values["WallMotionAbnormalities"])
            ]])

            # 3. scale using the training scaler
            features_scaled = scaler.transform(features)

            # 4. predict probability Disease=1
            proba = model.predict_proba(features_scaled)[0, 1]
            prob = round(float(proba) * 100, 1)

            # 5. map probability → text label
            if proba < 0.33:
                prediction = "Low estimated risk of cardiac disease"
            elif proba < 0.66:
                prediction = "Moderate estimated risk of cardiac disease"
            else:
                prediction = "High estimated risk of cardiac disease"

        except Exception as e:
            prediction = f"Could not compute risk (error: {e})"

    return render_template(
        "index.html",
        prediction=prediction,
        prob=prob,
        values=values
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

