from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import joblib
import numpy as np
from recommendations import provide_sleep_advice, predict_and_recommend_sleep
from qualitytraining import predict_optimal_sleep
import os
from dotenv import load_dotenv
from chatbot.gpt_response import get_gpt_response
from chatbot.database import fetch_health_advice

# Load environment variables from .env file (if using)
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "default_secret_key")  # Required for session
CORS(app)

gb_model_male = joblib.load('gb_model_male.pkl')
gb_model_female = joblib.load('gb_model_female.pkl')

sleep_quality_model = joblib.load('optimized_rf_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route("/")
def sleep_predictor_page():
    return render_template("sleep.html")

@app.route("/quality")
def quality():
    return render_template("quality.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

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
            "age", "gender", "sleep_duration", "caffeine_intake", "physical_activity"
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing one or more required fields"}), 400

        # Extract inputs
        age = float(data["age"])
        gender = data["gender"].strip().lower()
        sleep_duration = float(data["sleep_duration"])
        caffeine_intake = float(data["caffeine_intake"])
        physical_activity = float(data["physical_activity"])

        gender_binary = 1 if gender == "male" else 0

        # Prepare input data for the model
        #input_features = scaler.transform(np.array([[age, gender_binary, sleep_duration, study_hours, screen_time, caffeine_intake, physical_activity]]))

        # Predict sleep quality using the loaded model
        #predicted_quality = round(sleep_quality_model.predict(input_features)[0])

        real_quality, best_duration, best_quality, durations, qualities = predict_optimal_sleep(sleep_quality_model, scaler, age, gender_binary, sleep_duration, caffeine_intake, physical_activity)

        # Return the prediction result
        return jsonify({
            "predicted_sleep_quality": round(real_quality,2),
            "optimal_duration": round(best_duration,1),
            "optimal_quality": round(best_quality,2),
            "durations": durations.tolist(),  # Convert NumPy array to list for JSON serialization
            "qualities": qualities.tolist()   # Convert NumPy array to list for JSON serialization
        })

    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400

        # Initialize conversation history in session
        if "conversation_history" not in session:
            session["conversation_history"] = []

        # Handle advice queries
        if "advice" in user_message.lower():
            topic = user_message.lower().replace("advice", "").strip()
            advice = fetch_health_advice(topic)
            session["conversation_history"].append({"role": "assistant", "message": advice})
            return jsonify({"response": advice})

        # Otherwise, generate GPT response
        session["conversation_history"].append({"role": "user", "message": user_message})
        prompt = "\n".join([f"{entry['role'].capitalize()}: {entry['message']}" for entry in session["conversation_history"]])
        
        gpt_response = get_gpt_response(prompt, os.getenv("API_KEY"))
        session["conversation_history"].append({"role": "assistant", "message": gpt_response})
        return jsonify({"response": gpt_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)