# Cardiac Arrest Prediction in Newborn Babies 🚑👶

## Overview
This project aims to predict the likelihood of cardiac arrest in newborn babies using machine learning techniques. By analyzing various health parameters, the model provides insights that can help in early detection and intervention.

## Features
- **Machine Learning Model**: Utilizes predictive modeling for early detection of cardiac arrest.
- **Data Processing**: Cleans and preprocesses newborn health data for optimal model performance.
- **Database Integration**: Uses MongoDB for storing patient records and predictions.
- **API Integration**: Provides an API to interact with the prediction model.
- **Logging**: Keeps a record of all predictions and inputs for further analysis.
- **Visualization**: Includes tools to visualize the data and model predictions.

## Tools Used
- **Python** (Primary language for model development and API creation)
- **Flask** (For building the API)
- **MongoDB** (For storing patient and prediction data)
- **Pandas & NumPy** (For data processing and manipulation)
- **Scikit-Learn** (For machine learning model development)
- **Matplotlib & Seaborn** (For data visualization)
- **NLTK** (For text processing if needed)

## Requirements
Ensure you have the following dependencies installed before running the project:
- Python 3.x
- Flask
- pymongo
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn

## Installation
Clone the repository:
```bash
git clone https://github.com/meh-pavan/Cardiac-Arrest-Prediction.git
cd Cardiac-Arrest-Prediction
```

(Optional) Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the API server:
```bash
python api.py
```

Access the API and send patient data to get predictions.

## Database Logging
The system logs every prediction into MongoDB with the following details:
- **Patient ID**: Unique identifier for each patient
- **Timestamp**: When the prediction was made
- **Health Metrics**: Input data used for prediction
- **Prediction Result**: The output of the model

## Debugging & Visualization
- The system includes logging mechanisms to debug API requests and predictions.
- Data visualization tools help in understanding model performance and trends.

## License
This project is open-source and available under the MIT License.

## Contact
For any inquiries or issues, please reach out via email at your.email@example.com.

