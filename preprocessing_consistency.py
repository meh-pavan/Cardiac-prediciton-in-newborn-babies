# Compare training feature stats with a new sample
import numpy as np
import pickle

# Load scaler
with open('Models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Example raw patient data (before scaling)
raw_sample = np.array([[150, 92, 60, 45]])
scaled_sample = scaler.transform(raw_sample)

print("Raw sample:", raw_sample)
print("Scaled sample:", scaled_sample)
