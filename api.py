from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pickle
import logging
import os
from datetime import datetime, timedelta
from modules.online_learning import OnlineModel
from modules.xai_module import get_textual_explanation
from modules.alert_system import send_alert
from modules.database_module import get_db_manager
from stable_baselines3 import PPO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
MODEL_DIR = "models"
FEATURE_NAMES = ["heart_rate", "oxygen_sat", "blood_pressure", "respiration_rate"]

# Initialize FastAPI app
app = FastAPI(
    title="Newborn Cardiac Arrest Prediction API",
    description="API for predicting cardiac arrest risk in newborns",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Get database manager
db_manager = get_db_manager()

# Define request and response models
class PatientData(BaseModel):
    patient_id: str
    heart_rate: float
    oxygen_sat: float
    blood_pressure: float
    respiration_rate: float

class PredictionResponse(BaseModel):
    patient_id: str
    risk_score: float
    alert_triggered: bool
    explanation: Optional[str] = None
    alert_response: Optional[str] = None

class PatientHistoryResponse(BaseModel):
    patient_id: str
    history: List[dict]
    total_records: int

class PatientListResponse(BaseModel):
    patients: List[str]
    total: int

class DashboardStatsResponse(BaseModel):
    recent_predictions: List[dict]
    risk_distribution: dict
    risk_trend: List[dict]
    system_status: dict

# Load models at startup
@app.on_event("startup")
def load_models():
    global scaler, online_model, rl_agent
    
    try:
        # Load scaler
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully.")
        
        # Load online model
        online_model = OnlineModel()
        try:
            online_model.load_model(os.path.join(MODEL_DIR, 'online_model.pkl'))
            logger.info("Online model loaded successfully.")
        except Exception as e:
            logger.warning(f"Online model not found, starting fresh: {e}")
        
        # Load RL agent
        rl_agent = PPO.load(os.path.join(MODEL_DIR, 'rl_agent.zip'))
        logger.info("RL agent loaded successfully.")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

# Define endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the Newborn Cardiac Arrest Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    try:
        # Process patient data
        features = np.array([[
            data.heart_rate, 
            data.oxygen_sat, 
            data.blood_pressure, 
            data.respiration_rate
        ]])
        X_scaled = scaler.transform(features)
        
        # Get risk score
        risk_score = get_risk_score(online_model.model, X_scaled)
        logger.info(f"Risk score computed for {data.patient_id}: {risk_score:.2f}")
        
        # Use RL agent to decide on alert
        observation = np.array([risk_score]).reshape(1,)
        action, _ = rl_agent.predict(observation, deterministic=True)
        
        # Extract final action
        if isinstance(action, np.ndarray):
            try:
                final_action = int(action.item())
            except Exception:
                final_action = int(action[0])
        else:
            final_action = int(action)
        
        logger.info(f"RL agent action for {data.patient_id}: {final_action}")
        
        # Prepare response
        response = {
            "patient_id": data.patient_id,
            "risk_score": float(risk_score),
            "alert_triggered": final_action == 1
        }
        
        # Generate explanation if alert is triggered
        explanation = None
        if final_action == 1:
            try:
                explanation = get_textual_explanation(
                    online_model.model,
                    X_scaled,
                    feature_names=FEATURE_NAMES
                )
                response["explanation"] = explanation
                
                # Send alert
                alert_response = send_alert(data.patient_id, risk_score, explanation)
                response["alert_response"] = alert_response
                
            except Exception as e:
                logger.error(f"Error in explanation or alert: {e}")
                response["explanation"] = f"Explanation unavailable: {str(e)}"
        
        # Save prediction to database
        try:
            prediction_record = {
                "patient_id": data.patient_id,
                "timestamp": datetime.now().isoformat(),
                "heart_rate": float(data.heart_rate),
                "oxygen_sat": float(data.oxygen_sat),
                "blood_pressure": float(data.blood_pressure),
                "respiration_rate": float(data.respiration_rate),
                "risk_score": float(risk_score),
                "alert_triggered": final_action == 1,
                "explanation": explanation if explanation else ""
            }
            
            db_manager.insert_patient_data(prediction_record, collection_name="predictions")
            logger.info(f"Prediction saved to database for patient {data.patient_id}")
        except Exception as e:
            logger.error(f"Failed to save prediction to database: {e}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}/history", response_model=PatientHistoryResponse)
def get_patient_history(
    patient_id: str, 
    page: int = Query(1, ge=1), 
    page_size: int = Query(10, ge=1, le=100),
    time_filter: Optional[str] = Query(None, description="Time filter: 'week', 'month', 'all'")
):
    """Get patient prediction history from database."""
    try:
        # Calculate skip for pagination
        skip = (page - 1) * page_size
        
        # Set up time range filter if specified
        time_range = None
        if time_filter:
            now = datetime.now()
            if time_filter == 'week':
                time_range = {'start': now - timedelta(days=7), 'end': now}
            elif time_filter == 'month':
                time_range = {'start': now - timedelta(days=30), 'end': now}
        
        # Get total count first (for pagination info)
        query = {"patient_id": patient_id}
        if time_range:
            query["timestamp"] = {
                "$gte": time_range['start'].isoformat(),
                "$lte": time_range['end'].isoformat()
            }
        
        total_records = len(db_manager.get_patient_data(
            patient_id=patient_id, 
            collection_name="predictions",
            time_range=time_range
        ))
        
        # Then get paginated records
        records = db_manager.get_patient_data(
            patient_id=patient_id,
            collection_name="predictions",
            limit=page_size,
            skip=skip,
            time_range=time_range
        )
        
        # Convert ObjectId to string for each record
        for record in records:
            if '_id' in record:
                record['_id'] = str(record['_id'])
        
        return {
            "patient_id": patient_id,
            "history": records,
            "total_records": total_records
        }
    except Exception as e:
        logger.error(f"Error retrieving patient history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients", response_model=PatientListResponse)
def get_patients():
    """Get a list of all patients."""
    try:
        patient_ids = db_manager.get_all_patients(collection_name="predictions")
        return {
            "patients": patient_ids,
            "total": len(patient_ids)
        }
    except Exception as e:
        logger.error(f"Error retrieving patient list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/stats", response_model=DashboardStatsResponse)
def get_dashboard_stats():
    """Get statistics for the dashboard."""
    try:
        # Get recent predictions
        recent_predictions = db_manager.get_recent_data(collection_name="predictions", limit=10)
        for record in recent_predictions:
            if '_id' in record:
                record['_id'] = str(record['_id'])
        
        # Calculate risk distribution
        all_predictions = db_manager.get_patient_data(collection_name="predictions")
        risk_distribution = {
            "low": 0,
            "moderate": 0,
            "high": 0,
            "critical": 0
        }
        
        for pred in all_predictions:
            risk_score = pred.get('risk_score', 0)
            if risk_score >= 0.8:
                risk_distribution["critical"] += 1
            elif risk_score >= 0.6:
                risk_distribution["high"] += 1
            elif risk_score >= 0.4:
                risk_distribution["moderate"] += 1
            else:
                risk_distribution["low"] += 1
        
        # Calculate risk trend (last 7 days)
        now = datetime.now()
        risk_trend = []
        
        for i in range(6, -1, -1):
            day = now - timedelta(days=i)
            next_day = day + timedelta(days=1)
            
            day_predictions = db_manager.get_patient_data(
                collection_name="predictions",
                time_range={'start': day, 'end': next_day}
            )
            
            if day_predictions:
                avg_risk = sum(p.get('risk_score', 0) for p in day_predictions) / len(day_predictions)
            else:
                avg_risk = 0
                
            risk_trend.append({
                "date": day.strftime("%m/%d"),
                "avgRisk": avg_risk
            })
        
        # System status
        total_predictions = len(all_predictions)
        alerts_triggered = sum(1 for p in all_predictions if p.get('alert_triggered', False))
        
        system_status = {
            "model_status": "Online",
            "alert_system": "Active",
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_predictions": total_predictions,
            "alerts_triggered": alerts_triggered,
            "alert_percentage": (alerts_triggered / total_predictions * 100) if total_predictions > 0 else 0
        }
        
        return {
            "recent_predictions": recent_predictions,
            "risk_distribution": risk_distribution,
            "risk_trend": risk_trend,
            "system_status": system_status
        }
    except Exception as e:
        logger.error(f"Error retrieving dashboard stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to get risk score
def get_risk_score(model, X_new):
    """Calculate risk score from the model, handling different model types."""
    try:
        # Try predict_proba first (for classifiers)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_new)[0]
            # If binary classifier (2 classes)
            if len(probs) == 2:
                return probs[1]  # Probability of positive class
            else:
                # For multi-class, return highest probability
                return np.max(probs)
        
        # If no predict_proba, try decision_function (for some linear models)
        elif hasattr(model, 'decision_function'):
            decision = model.decision_function(X_new)[0]
            # Convert to probability-like score using sigmoid
            if isinstance(decision, np.ndarray):
                return 1 / (1 + np.exp(-np.max(decision)))
            else:
                return 1 / (1 + np.exp(-decision))
        
        # Last resort: use predict and assume binary output
        else:
            pred = model.predict(X_new)[0]
            # If prediction is already between 0 and 1, use it directly
            if 0 <= pred <= 1:
                return float(pred)
            else:
                # Otherwise, normalize to 0-1 range (assuming binary classification)
                return float(pred > 0)
    
    except Exception as e:
        logger.error(f"Error calculating risk score: {e}")
        # Return a high risk score by default to be safe
        return 1.0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)