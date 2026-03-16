from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and encoders
model = joblib.load('ev_model.pkl')
le_weekday = joblib.load('le_weekday.pkl')
le_platform = joblib.load('le_platform.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract features from request
        start_time = float(data['startTime'])
        end_time = float(data['endTime'])
        charge_hrs = float(data['chargeTimeHrs'])
        weekday = data['weekday']
        platform = data['platform']
        distance = float(data['distance'])
        station_id = int(data['stationId'])
        facility_type = int(data['facilityType'])

        # Encoding
        weekday_enc = le_weekday.transform([weekday])[0]
        platform_enc = le_platform.transform([platform])[0]

        # Final Feature Array
        features = np.array([[start_time, end_time, charge_hrs, weekday_enc, 
                              platform_enc, distance, station_id, facility_type]])
        
        prediction = model.predict(features)[0]
        
        return jsonify({'prediction': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)