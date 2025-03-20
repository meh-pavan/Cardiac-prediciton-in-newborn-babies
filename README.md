# Newborn Cardiac Arrest Prediction System

This system uses machine learning to predict the risk of cardiac arrest in newborn babies, providing real-time alerts and explanations for medical staff.

## Features

- **Risk Prediction**: Analyzes vital signs to predict cardiac arrest risk
- **Explainable AI**: Provides explanations for predictions
- **Alert System**: Sends alerts to medical staff for high-risk cases
- **Online Learning**: Continuously improves the model with new data
- **Reinforcement Learning**: Optimizes alert thresholds
- **Web Dashboard**: Visualizes predictions and patient history
- **Database Integration**: Stores all predictions for historical analysis

## Project Structure

- **Backend**: Python-based ML pipeline with FastAPI
- **Frontend**: React-based web interface
- **Models**: Pre-trained ML models and RL agents
- **Database**: MongoDB Atlas for storing prediction data

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd newborn-cardiac-arrest-prediction
```

### 2. Set Up the Database

This project uses MongoDB Atlas with the following connection string:

```
mongodb+srv://pavanmacha46:qwer1234@cluster0.smjhu.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
```

The connection string is already configured in the `.env` file.

### 3. Set Up the Backend

#### Create and Activate a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### Install Backend Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt file, install these packages:

```bash
pip install fastapi uvicorn numpy pandas scikit-learn pymongo python-dotenv twilio stable-baselines3 shap
```

#### Test the Database Connection

```bash
python test_db_connection.py
```

### 4. Set Up the Frontend

#### Navigate to the Frontend Directory

```bash
cd cardiac-prediction-frontend
```

#### Install Frontend Dependencies

```bash
npm install
```

Note: The package.json file contains some version numbers that might be too new. If you encounter errors, update the package.json with these compatible versions:

```json
"dependencies": {
  "@emotion/react": "^11.11.0",
  "@emotion/styled": "^11.11.0",
  "@mui/icons-material": "^5.14.0",
  "@mui/material": "^5.14.0",
  "@testing-library/jest-dom": "^5.17.0",
  "@testing-library/react": "^13.4.0",
  "@testing-library/user-event": "^13.5.0",
  "axios": "^1.6.2",
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-router-dom": "^6.20.0",
  "react-scripts": "5.0.1",
  "recharts": "^2.10.1",
  "web-vitals": "^2.1.4"
}
```

Then run `npm install` again.

### 5. Start the Application

#### Start the Backend Server

In the main project directory (with the virtual environment activated):

```bash
python api.py
```

The API will be available at http://localhost:8000

#### Start the Frontend Development Server

In a new terminal, navigate to the frontend directory:

```bash
cd cardiac-prediction-frontend
npm start
```

The web interface will be available at http://localhost:3000

## Using the Application

1. Open the web interface at http://localhost:3000
2. Navigate to "New Prediction" to enter patient data
3. Use the following examples for testing:

### High Risk Example
```
Patient ID: HIGH001
Heart Rate: 180 bpm
Oxygen Saturation: 85%
Blood Pressure: 45 mmHg
Respiration Rate: 70 breaths/min
```

### Low Risk Example
```
Patient ID: LOW001
Heart Rate: 120 bpm
Oxygen Saturation: 98%
Blood Pressure: 60 mmHg
Respiration Rate: 40 breaths/min
```

4. View prediction results and explanations
5. Check the dashboard for an overview of recent predictions
6. View patient history for detailed prediction records

## API Endpoints

- `GET /`: API welcome message
- `POST /predict`: Submit patient data for prediction
- `GET /patients/{patient_id}/history`: Get patient prediction history
- `GET /patients`: Get a list of all patients
- `GET /dashboard/stats`: Get statistics for the dashboard

## Troubleshooting

### Backend Issues

- **MongoDB Connection Error**: Verify internet connectivity and that the MongoDB Atlas connection string is correct
- **Missing Dependencies**: Ensure all required Python packages are installed
- **Port Already in Use**: Make sure no other application is using port 8000

### Frontend Issues

- **Dependency Errors**: If you see errors about React version compatibility, update the package.json file with the compatible versions listed above
- **Connection to API Failed**: Ensure the backend server is running and accessible at http://localhost:8000
- **CORS Issues**: Make sure the backend has CORS enabled for the frontend origin

### Common Solutions

- **Restart Servers**: Sometimes simply restarting both the frontend and backend servers can resolve issues
- **Clear Browser Cache**: Try clearing your browser cache or using incognito mode
- **Check Console Logs**: Look at both the terminal running the servers and the browser console for error messages

## Security Note

The MongoDB connection string in this README contains actual credentials. For security:

1. Change your MongoDB Atlas password after completing this project
2. Never commit credentials to version control
3. Consider using environment variables for sensitive information in production

## License

This project is licensed under the MIT License - see the LICENSE file for details.