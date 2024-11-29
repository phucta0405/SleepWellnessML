from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from recommendations import provide_sleep_advice, predict_and_recommend_sleep
from qualitytraining import scaler

app = Flask(__name__)

gb_model_male = joblib.load('gb_model_male.pkl')
gb_model_female = joblib.load('gb_model_female.pkl')

sleep_quality_model = joblib.load('gb_model_quality.pkl')

@app.route("/")
def sleep_predictor_page():
    return render_template("sleep.html")

@app.route("/quality")
def quality():
    return render_template("quality.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    required_fields = ["age", "gender", "physical_activity_minutes"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing one or more required fields: age, gender, physical_activity_minutes"}), 400

    try:
        age = float(data["age"])
        gender = data["gender"].strip()
        physical_activity_minutes = float(data["physical_activity_minutes"])
        actual_sleep_duration = float(data["sleep_duration"])
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

@app.route("/predict-sleep-quality", methods=["POST"])
def predict_sleep_quality():
    try:
        # Parse the JSON data from the request
        data = request.json

        # Validate required fields
        required_fields = [
            "age", "gender", "sleep_duration", "study_hours",
            "screen_time", "caffeine_intake", "physical_activity"
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Extract inputs
        age = float(data["age"])
        gender = data["gender"].strip().lower()
        sleep_duration = float(data["sleep_duration"])
        study_hours = float(data["study_hours"])
        screen_time = float(data["screen_time"])
        caffeine_intake = float(data["caffeine_intake"])
        physical_activity = float(data["physical_activity"])

        gender_binary = 1 if gender == "male" else 0

        # Prepare input data for the model
        input_features = scaler.transform(np.array([[age, gender_binary, sleep_duration, study_hours, screen_time, caffeine_intake, physical_activity]]))

        # Predict sleep quality using the loaded model
        predicted_quality = round(sleep_quality_model.predict(input_features)[0])

        # Return the prediction result
        return jsonify({
            "predicted_sleep_quality": predicted_quality
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)