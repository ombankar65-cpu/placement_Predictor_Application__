from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "KNN Placement Prediction API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the correct order as per model metadata
        # features: [gender, ssc_p, hsc_p, hsc_s, degree_p, mba_p]
        features = [
            data['gender'],
            data['ssc_p'],
            data['hsc_p'],
            data['hsc_s'],
            data['degree_p'],
            data['mba_p']
        ]
        
        # Convert to numpy array and reshape for prediction
        final_features = np.array([features])
        prediction = model.predict(final_features)
        
        return jsonify({
            'prediction': str(prediction[0])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
