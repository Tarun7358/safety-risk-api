from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load ML Model
# -----------------------------
model = joblib.load("model.pkl")

# -----------------------------
# Load Firebase Firestore (optional)
# -----------------------------
cred = credentials.Certificate("firebase_admin.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# Environment Variables (Render)
# -----------------------------
FAST2SMS_KEY = os.getenv("FAST2SMS_KEY")
EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER", "+919840595720")

# -----------------------------
# Predict Risk
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        data["age"],
        data["timeOfDay"],
        data["crowdDensity"],
        data["areaSafetyScore"],
        data["weather"]
    ]])

    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0].tolist()

    result = {
        "prediction": int(pred),
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

# -----------------------------
# Panic SOS Alert (SMS)
# -----------------------------
@app.route("/panic", methods=["POST"])
def panic():
    data = request.json

    message = f"🚨 SOS ALERT!\nUser: {data['name']}\nLat: {data['lat']}\nLng: {data['lng']}"

    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "message": message,
        "route": "q",
        "numbers": EMERGENCY_NUMBER
    }

    headers = {
        "authorization": FAST2SMS_KEY,
        "Content-Type": "application/json"
    }

    r = requests.post(url, json=payload, headers=headers)

    return jsonify({"status": "sent", "fast2sms_response": r.json()})

# -----------------------------
# Save Location
# -----------------------------
@app.route("/update_location", methods=["POST"])
def update_location():
    data = request.json

    db.collection("locations").document(data["email"]).set({
        "lat": data["lat"],
        "lng": data["lng"],
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    return jsonify({"status": "location updated"})

# -----------------------------
# Fetch Last Prediction
# -----------------------------
@app.route("/last/<email>")
def last_prediction(email):
    docs = db.collection("predictions").where("email", "==", email).order_by(
        "timestamp", direction=firestore.Query.DESCENDING).limit(1).stream()

    for doc in docs:
        return jsonify(doc.to_dict())

    return jsonify({"error": "no predictions found"})

# -----------------------------
# Root Endpoint
# -----------------------------
@app.route("/")
def home():
    return jsonify({"message": "Safety Risk API Running!"})

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
