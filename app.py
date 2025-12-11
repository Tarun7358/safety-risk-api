from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load Model
model = joblib.load("model.pkl")

# Firebase Initialization
cred = credentials.Certificate("firebase_admin.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ENV Variables
FAST2SMS_KEY = os.getenv("FAST2SMS_KEY")
EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER")


# ============================================
# 🔵 1. RISK PREDICTION API
# ============================================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Create feature vector
    features = np.array([[ 
        data["age"],
        data["timeOfDay"],
        data["crowdDensity"],
        data["areaSafetyScore"],
        data["weather"]
    ]])

    # Predictions
    pred = int(model.predict(features)[0])
    probs = model.predict_proba(features)[0].tolist()

    result = {
        "prediction": pred,
        "probabilities": {
            "low": probs[0],
            "medium": probs[1],
            "high": probs[2]
        }
    }

    # Save to Firestore
    db.collection("predictions").add({
        "email": data["email"],
        "result": result,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return jsonify(result)


# ============================================
# 🔴 2. PANIC ALERT API (SMS)
# ============================================
@app.route("/panic", methods=["POST"])
def panic():
    data = request.json
    name = data["name"]
    lat = data["lat"]
    lng = data["lng"]

    msg = f"🚨 EMERGENCY ALERT!\n{name} may be in danger.\nLocation: https://maps.google.com/?q={lat},{lng}"

    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "message": msg,
        "language": "english",
        "route": "q",
        "numbers": EMERGENCY_NUMBER
    }

    headers = {
        "authorization": FAST2SMS_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)
    return jsonify({"status": "sent", "sms_response": response.json()})


# ============================================
# 📍 3. SAVE LIVE LOCATION API
# ============================================
@app.route("/update_location", methods=["POST"])
def update_location():
    data = request.json

    db.collection("locations").document(data["email"]).set({
        "lat": data["lat"],
        "lng": data["lng"],
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return jsonify({"status": "location_saved"})


# ============================================
# ▶ START SERVER
# ============================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
