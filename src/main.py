from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model & scaler
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI(title="Fraud Detection API")

# Input schema
class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict_fraud(data: Transaction):
    input_data = np.array([[value for value in data.dict().values()]])
    input_scaled = scaler.transform(input_data)

    prob = model.predict_proba(input_scaled)[0][1]
    risk_score = round(prob * 100, 2)

    if risk_score >= 80:
        decision = "HIGH RISK"
    elif risk_score >= 50:
        decision = "MEDIUM RISK"
    else:
        decision = "LOW RISK"

    return {
        "fraud_probability": round(prob, 4),
        "risk_score": risk_score,
        "decision": decision
    }
