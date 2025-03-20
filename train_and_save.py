import numpy as np
import pickle
import logging
from modules.data_module import load_synthetic_data
from modules.preprocessing import preprocess_data
from modules.model_training import train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    try:
        # Load synthetic data (set plot=False for training)
        logging.info("Loading synthetic data...")
        df = load_synthetic_data(plot=False)
        logging.info("Synthetic data loaded successfully.")
        
        # Log label distribution for verification
        logging.info("Label distribution:\n%s", df['label'].value_counts().to_string())
        
        # Data Preprocessing: extract features and apply scaling
        logging.info("Preprocessing data...")
        features_scaled, labels, scaler = preprocess_data(df)
        logging.info("Data preprocessing completed.")
        
        # Model Training
        logging.info("Training the model...")
        model, X_test, y_test = train_model(features_scaled, labels)
        logging.info("Model training completed.")
        
        # Save the trained model and scaler to disk
        logging.info("Saving model and scaler...")
        with open('Models/model.pkl', 'wb') as model_file:
            pickle.dump(model, model_file)
        with open('Models/scaler.pkl', 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)
        logging.info("Model and scaler have been saved successfully.")
        
    except Exception as e:
        logging.error("An error occurred during training and saving.", exc_info=True)

if __name__ == '__main__':
    main()
