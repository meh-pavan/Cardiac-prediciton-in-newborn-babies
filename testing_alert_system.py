from modules.alert_system import send_alert

# Sample test data
patient_id = "TestPatient001"   
risk_score = 0.85
explanation = "Top contributing features: heart_rate: 2.45, oxygen_sat: -1.10, blood_pressure: 0.50"

# Send a test alert
response = send_alert(patient_id, risk_score, explanation)
print("Alert response:", response)
