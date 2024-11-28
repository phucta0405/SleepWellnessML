from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from recommendations import provide_sleep_advice, predict_and_recommend_sleep

app = Flask(__name__)

gb_model_male = joblib.load('gb_model_male.pkl')
gb_model_female = joblib.load('gb_model_female.pkl')

@app.route("/")
def sleep_predictor_page():
    return render_template("sleep.html")

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
        sleep_advice = provide_sleep_advice(actual_sleep_duration, age, gender, physical_activity_minutes)
        return jsonify({
            "predicted_sleep_duration": round(predicted_sleep_duration, 2),
            "sleep_advice": sleep_advice
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(debug=True)