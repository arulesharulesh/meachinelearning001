from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Define the path to the model file. 
# Ensure 'logistic_regression_model.joblib' is in the same directory as app.py
# or provide an absolute path.
MODEL_PATH = 'logistic_regression_model.joblib'

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file not found at {MODEL_PATH}")

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API! Use /predict for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded or not found.'}), 500

    try:
        data = request.get_json(force=True)

        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, list):
            input_df = pd.DataFrame(data)
        else:
            return jsonify({'error': 'Invalid input data format. Expected dict or list of dicts.'}), 400

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        prediction_list = prediction.tolist()
        proba_list = prediction_proba.tolist()

        return jsonify({
            'prediction': prediction_list,
            'prediction_proba': proba_list
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For local execution, you can run this file directly:
    # python app.py
    # It will typically run on http://127.0.0.1:5000
    app.run(host='0.0.0.0', port=5000, debug=False)
