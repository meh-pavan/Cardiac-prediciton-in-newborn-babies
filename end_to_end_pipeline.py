import numpy as np
import pickle
import logging
import pandas as pd
import os
from modules.online_learning import OnlineModel
from modules.xai_module import get_textual_explanation
from modules.alert_system import send_alert
from stable_baselines3 import PPO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
MODEL_DIR = "models"  # Lowercase to match project structure
FEATURE_NAMES = ["heart_rate", "oxygen_sat", "blood_pressure", "respiration_rate"]

def load_scaler(filepath=os.path.join(MODEL_DIR, 'scaler.pkl')):
    """Load the scaler from disk."""
    try:
        with open(filepath, 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Scaler loaded successfully.")
        return scaler
    except FileNotFoundError:
        logger.error(f"Scaler file not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        raise

def load_online_model(filepath=os.path.join(MODEL_DIR, 'online_model.pkl')):
    """Initialize and load the online model from disk, or start fresh if not found."""
    online_model = OnlineModel()
    try:
        online_model.load_model(filepath)
        logger.info("Online model loaded from disk.")
    except FileNotFoundError:
        logger.warning("Online model not found, starting fresh.")
    except Exception as e:
        logger.error(f"Error loading online model: {e}")
        raise
    return online_model

def load_rl_agent(filepath=os.path.join(MODEL_DIR, 'rl_agent.zip')):
    """Load the trained RL agent from disk."""
    try:
        rl_agent = PPO.load(filepath)
        logger.info("RL agent loaded from disk.")
        return rl_agent
    except FileNotFoundError:
        logger.error(f"RL agent file not found at {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading RL agent: {e}")
        raise

def process_new_patient_data(patient_data, scaler):
    """
    Convert a dictionary of patient data to a DataFrame and preprocess it using the provided scaler.
    
    Parameters:
        patient_data (dict): Patient vital signs.
        scaler: The scaler to transform the data.
        
    Returns:
        np.array: Preprocessed feature array.
    """
    try:
        df = pd.DataFrame([patient_data])
        features = df[FEATURE_NAMES]
        X_new = scaler.transform(features)
        return X_new
    except Exception as e:
        logger.error(f"Error processing patient data: {e}")
        raise

def get_risk_score(model, X_new):
    """
    Calculate risk score from the model, handling different model types.
    
    Parameters:
        model: The trained model
        X_new: Preprocessed input features
        
    Returns:
        float: Risk score between 0 and 1
    """
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

def main():
    try:
        # Step 1: Load required artifacts
        scaler = load_scaler()
        online_model = load_online_model()
        rl_agent = load_rl_agent()

        # Step 2: Simulate new patient data input (this would normally come from your database or frontend)
        new_patient_data = {
            'heart_rate': 150,         # Example values; adjust as needed
            'oxygen_sat': 92,
            'blood_pressure': 60,
            'respiration_rate': 45
        }
        patient_id = "Patient001"

        # Process new patient data
        X_new = process_new_patient_data(new_patient_data, scaler)
        logger.info("New patient data processed.")

        # Step 3: Get risk score from the online classifier using the robust method
        risk_score = get_risk_score(online_model.model, X_new)
        logger.info(f"Risk score computed: {risk_score:.2f}")

        # Step 4: Use the RL agent to decide on alert action.
        observation = np.array([risk_score]).reshape(1,)
        action, _ = rl_agent.predict(observation, deterministic=True)
        
        # Robust extraction of final_action
        if isinstance(action, np.ndarray):
            try:
                final_action = int(action.item())
            except Exception:
                final_action = int(action[0])
        else:
            final_action = int(action)
        logger.info("RL agent action: %d", final_action)

        # Step 5: If the RL agent decides to trigger an alert, generate explanation and send alert.
        if final_action == 1:
            # Generate explanation with better error handling
            try:
                explanation = get_textual_explanation(
                    online_model.model, 
                    X_new, 
                    feature_names=FEATURE_NAMES
                )
                logger.info("Explanation generated: %s", explanation)
            except Exception as e:
                logger.error(f"Error generating explanation: {e}")
                explanation = "Explanation unavailable due to technical error."
            
            # Send alert
            try:
                alert_response = send_alert(patient_id, risk_score, explanation)
                logger.info("Alert response: %s", alert_response)
            except Exception as e:
                logger.error(f"Error sending alert: {e}")
        else:
            logger.info("No alert triggered.")
    
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == '__main__':
    main()