from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="Safety Risk Prediction API")

# Input format expected from Flutter
class RiskInput(BaseModel):
    locationType: str
    activity: str
    time: str
    weather: str
    heartRate: int

# Rule-based fallback model (offline)
def rule_engine(data):
    score = 0
    factors = []

    # Time factor
    if data.time.lower() == "night":
        score += 25
        factors.append("Night Time")

    # Weather factor
    if data.weather.lower() in ["rain", "storm", "fog"]:
        score += 20
        factors.append("Bad Weather")

    # Activity factor
    if data.activity.lower() in ["driving fast", "running"]:
        score += 20
        factors.append("High-risk activity")

    # Heart rate
    if data.heartRate > 110:
        score += 15
        factors.append("High heart rate")

    # Location
    if data.locationType.lower() in ["street", "isolated"]:
        score += 20
        factors.append("Unsafe location")

    # Determine level
    level = "Low"
    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Medium"

    return {
        "risk_level": level,
        "risk_score": score,
        "factors": factors
    }


@app.get("/")
def root():
    return {"message": "Safety Risk Prediction API is Running!"}


@app.post("/ml/predict")
def predict_risk(data: RiskInput):
    result = rule_engine(data)
    return result


@app.post("/ml/insights")
def insights(data: RiskInput):
    insights_list = []

    if data.time.lower() == "night":
        insights_list.append("Risk increases significantly during night.")

    if data.weather.lower() in ["rain", "storm", "fog"]:
        insights_list.append("Bad weather reduces visibility and increases risk.")

    if data.heartRate > 110:
        insights_list.append("Elevated heart rate indicates stress or danger.")

    if not insights_list:
        insights_list.append("No major risk contributors detected.")

    return {
        "insights": insights_list
    }


@app.get("/ml/offlineRules")
def offline_rules():
    return {
        "rules": [
            "Night increases risk by +25",
            "Stormy weather increases risk by +20",
            "High heart rate increases risk by +15",
            "Isolated areas increase risk by +20"
        ]
    }
