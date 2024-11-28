from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained models
sleep_duration_model = joblib.load("sleep_duration_model.pkl")  # Replace with your model file
sleep_quality_model = joblib.load("sleep_quality_model.pkl")    # Replace with your model file

# Route for the Sleep Predictor Page
@app.route("/")
def sleep_predictor_page():
    return render_template("predictor.html")  # Serve the HTML page (place this file in the templates/ folder)

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the request
    data = request.json

    # Validate input fields
    required_fields = ["age", "gender", "physical_activity_minutes"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing one or more required fields: age, gender, physical_activity_minutes"}), 400

    try:
        age = float(data["age"])
        gender = data["gender"].strip().lower()
        physical_activity_minutes = float(data["physical_activity_minutes"])

        # Encode gender: Male = 1, Female = 0
        gender_encoded = 1 if gender == "male" else 0

        # Prepare input features for models
        features = np.array([[age, gender_encoded, physical_activity_minutes]])

        # Make predictions
        sleep_duration = sleep_duration_model.predict(features)[0]
        sleep_quality = sleep_quality_model.predict(features)[0]

        # Map sleep quality to a more descriptive string (if needed)
        sleep_quality_mapping = {0: "Poor", 1: "Average", 2: "Good"}
        sleep_quality_label = sleep_quality_mapping.get(int(sleep_quality), "Unknown")

        # Return predictions
        return jsonify({
            "sleep_duration": round(sleep_duration, 2),
            "sleep_quality": sleep_quality_label
        })

    except ValueError:
        return jsonify({"error": "Invalid input values. Ensure all inputs are numeric where required."}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
