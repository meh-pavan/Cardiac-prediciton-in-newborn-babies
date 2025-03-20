import numpy as np
import pandas as pd
import pickle
import logging
from modules.preprocessing import preprocess_data  # Not used directly here, but may be useful for future enhancements
from modules.online_learning import OnlineModel

# Configure logging to include timestamp and log level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_new_data():
    """
    Simulate retrieval of new data for online model updates.
    
    Returns:
        df_new (pd.DataFrame): A new synthetic DataFrame containing vital sign data and labels.
    """
    np.random.seed(100)  # Use a different seed for new data generation
    n_samples = 200

    # Generate synthetic vital signs with realistic ranges
    heart_rate = np.random.randint(120, 161, size=n_samples)          # 120-160 bpm
    oxygen_sat = np.random.randint(90, 101, size=n_samples)             # 90-100%
    blood_pressure = np.random.randint(50, 71, size=n_samples)          # 50-70 mmHg
    respiration_rate = np.random.randint(30, 61, size=n_samples)        # 30-60 breaths per minute

    # Create labels using the same rule as training:
    # If heart_rate > 140 and oxygen_sat < 95 then label = 1, else 0.
    labels = []
    for hr, ox in zip(heart_rate, oxygen_sat):
        labels.append(1 if hr > 140 and ox < 95 else 0)
    labels = np.array(labels)

    # Construct a DataFrame with the generated data
    df_new = pd.DataFrame({
        'heart_rate': heart_rate,
        'oxygen_sat': oxygen_sat,
        'blood_pressure': blood_pressure,
        'respiration_rate': respiration_rate,
        'label': labels
    })

    logging.info("New synthetic data generated with %d samples.", n_samples)
    return df_new

def update_online_model():
    """
    Load the existing scaler and online model, update the model with new data, 
    and save the updated model to disk.
    """
    try:
        # Load the scaler from disk (used during training for data preprocessing)
        logging.info("Loading scaler from disk...")
        with open('Models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logging.info("Scaler loaded successfully.")

        # Initialize the online model instance
        online_model = OnlineModel()
        try:
            # Try to load an existing online model from disk
            online_model.load_model('Models/online_model.pkl')
            logging.info("Loaded existing online model.")
        except FileNotFoundError:
            logging.info("No existing online model found; starting fresh.")

        # Retrieve new synthetic data
        logging.info("Fetching new data for online model update...")
        df_new = get_new_data()
        features_new = df_new[['heart_rate', 'oxygen_sat', 'blood_pressure', 'respiration_rate']]
        labels_new = df_new['label']

        # Preprocess new features using the loaded scaler
        X_new = scaler.transform(features_new)
        logging.info("New data preprocessed successfully.")

        # Update the online model incrementally with new data
        logging.info("Updating online model with new data...")
        online_model.partial_train(X_new, labels_new)

        # Save the updated online model back to disk
        with open('Models/online_model.pkl', 'wb') as f:
            pickle.dump(online_model.model, f)
        logging.info("Online model updated and saved successfully.")

    except Exception as e:
        logging.error("An error occurred while updating the online model.", exc_info=True)

if __name__ == '__main__':
    update_online_model()
