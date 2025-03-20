from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import logging
from stable_baselines3 import PPO

# Import modules from your project
from modules.online_learning import OnlineModel
from modules.xai_module import get_textual_explanation
from modules.alert_system import send_alert

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Newborn Cardiac Arrest Prediction API",
    description="An API to process patient data, compute risk scores, and trigger alerts based on RL decisions with explainable AI.",
    version="1.0.0"
)

# Enable CORS for all origins (adjust as necessary)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request body schema using Pydantic
class PatientData(BaseModel):
    patient_id: str = "Unknown"
    heart_rate: float
    oxygen_sat: float
    blood_pressure: float
    respiration_rate: float

# Load artifacts at startup

# Load scaler
try:
    with open('Models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    logging.info("Scaler loaded successfully.")
except Exception as e:
    logging.error("Error loading scaler: %s", e)
    scaler = None

# Initialize and load online model
online_model = OnlineModel()
try:
    online_model.load_model('Models/online_model.pkl')
    logging.info("Online model loaded successfully.")
except Exception as e:
    logging.info("No existing online model found; starting fresh.")

# Load the RL agent
try:
    rl_agent = PPO.load("Models/rl_agent.zip")
    logging.info("RL agent loaded successfully.")
except Exception as e:
    logging.error("Error loading RL agent: %s", e)
    rl_agent = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Newborn Cardiac Arrest Prediction API"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input data to a NumPy array
        features = np.array([[data.heart_rate, data.oxygen_sat, data.blood_pressure, data.respiration_rate]])
        logging.info("Received patient data: %s", features)

        # Ensure scaler is loaded and scale the features
        if scaler is None:
            raise HTTPException(status_code=500, detail="Scaler not loaded.")
        features_scaled = scaler.transform(features)
        logging.info("Features scaled: %s", features_scaled)

        # Compute risk score from the online model
        risk_probabilities = online_model.model.predict_proba(features_scaled)[0]
        risk_score = risk_probabilities[1]  # Assume index 1 corresponds to high risk
        logging.info("Risk score computed: %.2f", risk_score)

        # Use the RL agent to decide whether to trigger an alert
        if rl_agent is None:
            raise HTTPException(status_code=500, detail="RL agent not loaded.")
        observation = np.array([risk_score]).reshape(1,)
        action, _ = rl_agent.predict(observation, deterministic=True)
        if isinstance(action, np.ndarray):
            try:
                final_action = int(action.item())
            except Exception:
                final_action = int(action[0])
        else:
            final_action = int(action)
        logging.info("RL agent action: %d", final_action)

        # If the RL agent decides to trigger an alert (action == 1)
        if final_action == 1:
            explanation = get_textual_explanation(
                online_model.model,
                features_scaled,
                feature_names=["heart_rate", "oxygen_sat", "blood_pressure", "respiration_rate"]
            )
            logging.info("Explanation generated: %s", explanation)
            alert_response = send_alert(data.patient_id, risk_score, explanation)
            logging.info("Alert response: %s", alert_response)
            return {
                "risk_score": risk_score,
                "alert": True,
                "explanation": explanation,
                "alert_response": alert_response
            }
        else:
            return {"risk_score": risk_score, "alert": False}

    except Exception as e:
        logging.error("Error during prediction: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
