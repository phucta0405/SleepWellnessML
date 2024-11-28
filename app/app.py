from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from recommendations import provide_sleep_advice, predict_and_recommend_sleep

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained models
gb_model_male = joblib.load('gb_model_male.pkl')
gb_model_female = joblib.load('gb_model_female.pkl')

# Route for the Sleep Predictor Page
@app.route("/")
def sleep_predictor_page():
    return render_template("sleep.html")  # Serve the HTML page (place this file in the templates/ folder)

# API Endpoint for Predictions and Recommendations
@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the request
    data = request.json

    # Validate input fields
    required_fields = ["age", "gender", "physical_activity_minutes"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing one or more required fields: age, gender, physical_activity_minutes"}), 400

    try:
        # Extract inputs
        age = float(data["age"])
        gender = data["gender"].strip()
        physical_activity_minutes = float(data["physical_activity_minutes"])
        actual_sleep_duration = float(data["sleep_duration"])

        # Predict sleep duration using the models
        predicted_sleep_duration = predict_and_recommend_sleep(age, gender, physical_activity_minutes)

        # Generate sleep advice
        sleep_advice, isEnough = provide_sleep_advice(actual_sleep_duration, age, gender, physical_activity_minutes)

        # Return predictions and advice
        return jsonify({
            "predicted_sleep_duration": round(predicted_sleep_duration, 2),
            "sleep_advice": sleep_advice,
            "isEnough": isEnough
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)