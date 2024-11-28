import joblib

gb_model_male = joblib.load('gb_model_male.pkl')
gb_model_female = joblib.load('gb_model_female.pkl')

def predict_and_recommend_sleep(user_age, gender_input, physical_activities):
    if gender_input not in ["Male", "Female"]:
        print("Invalid gender input. Please input 'Male' or 'Female'.")
        return

    if gender_input == "Male":
        predicted_sleep_duration = gb_model_male.predict([[user_age]])[0]
    else:
        predicted_sleep_duration = gb_model_female.predict([[user_age]])[0]

    advice = provide_sleep_advice(predicted_sleep_duration, user_age, gender_input)

    print(f"Predicted Sleep Duration for Age {user_age} ({gender_input}): {predicted_sleep_duration:.2f} hours")
    print(advice)
