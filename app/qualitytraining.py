import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../data/student_sleep_patterns.csv")
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data = data.dropna()
X = data[["Age", "Gender", "Sleep_Duration", "Study_Hours", "Screen_Time", "Caffeine_Intake", "Physical_Activity"]]
Y = data[["Sleep_Quality"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

gb_model = GradientBoostingRegressor(random_state=42)


gb_model.fit(X_train, y_train)


joblib.dump(gb_model, 'gb_model_quality.pkl')



"""
try:
    user_age = float(input("Input your age: "))
    gender_input = input("Input your gender (Male/Female): ").strip()
    physical_activity = float(input("Input your level of Physical Activity (minutes/day): "))
    if gender_input not in ["Male", "Female"]:
        print("Invalid gender input. Please input 'Male' or 'Female'.")
    else:
        # Encode gender as a dummy variable
        gender_male = 1 if gender_input == "Male" else 0
        predicted_sleep_duration = gb_model.predict([[user_age, gender_male, physical_activity]])[0]
        print(f"Predicted Sleep Duration for Age {user_age} ({gender_input}): {predicted_sleep_duration:.2f} hours")
except ValueError:
    print("Invalid age input. Please input a numeric value.")
"""