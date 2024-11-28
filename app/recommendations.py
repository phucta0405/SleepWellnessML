import joblib

gb_model_male = joblib.load('gb_model_male.pkl')
gb_model_female = joblib.load('gb_model_female.pkl')

def provide_sleep_advice(predicted_sleep_duration, age, gender_input, physical_activities):
    recommended_sleep = round(predict_and_recommend_sleep(age, gender_input, physical_activities),2)

    if predicted_sleep_duration < recommended_sleep:
        return f"At age {age}, it's recommended to get around {recommended_sleep} hours of sleep. You should aim to improve your sleep duration for better wellness."
    elif predicted_sleep_duration > recommended_sleep:
        return f"You're getting more than the recommended sleep duration of {recommended_sleep} hours. Ensure it's quality sleep, but be mindful of not oversleeping."
    else:
        return f"You're on track with the recommended sleep duration of {recommended_sleep} hours for your age. Keep up the good work!"

def predict_and_recommend_sleep(user_age, gender_input, physical_activities):
    if gender_input not in ["Male", "Female"]:
        print("Invalid gender input. Please input 'Male' or 'Female'.")
        return 0 
    input_data = [[user_age, physical_activities]]

    if gender_input == "Male":
        predicted_sleep_duration = gb_model_male.predict(input_data)[0]
    else:
        predicted_sleep_duration = gb_model_female.predict(input_data)[0]

    return predicted_sleep_duration

def check_sleep_goal(sleep_duration, recommended_sleep_duration):
    if sleep_duration < recommended_sleep_duration:
        return f"You're not meeting the recommended sleep goal. Try to aim for at least {recommended_sleep_duration:.2f} hours of sleep."
    elif sleep_duration > recommended_sleep_duration:
        return f"You're getting more than the recommended sleep duration of {recommended_sleep_duration:.2f} hours. Make sure it's quality sleep, but avoid oversleeping."
    else:
        return f"You're on track! You're meeting the recommended sleep duration of {recommended_sleep_duration:.2f} hours."


