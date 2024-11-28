import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("../data/Sleep_health_and_lifestyle_dataset.csv")
data1 = pd.read_csv("../data/student_sleep_patterns.csv")

data.rename(columns={"Sleep Duration": "Sleep_Duration", "Physical Activity Level": "Physical_Activity"}, inplace=True)

data_filtered = data[["Age", "Sleep_Duration", "Gender", "Physical_Activity"]]
data1_filtered = data1[["Age", "Sleep_Duration", "Gender", "Physical_Activity"]]

combined_data = pd.concat([data_filtered, data1_filtered], ignore_index=True)

combined_data.dropna(inplace=True)

combined_data = pd.get_dummies(combined_data, columns=["Gender"], drop_first=True)

male_data = combined_data[combined_data["Gender_Male"] == 1]
female_data = combined_data[combined_data["Gender_Male"] == 0]

X_male = male_data[["Age", "Physical_Activity"]]
y_male = male_data["Sleep_Duration"]

X_female = female_data[["Age", "Physical_Activity"]]
y_female = female_data["Sleep_Duration"]

X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(X_male, y_male, test_size=0.2, random_state=42)
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(X_female, y_female, test_size=0.2, random_state=42)

gb_model_male = GradientBoostingRegressor(random_state=42)
gb_model_female = GradientBoostingRegressor(random_state=42)

gb_model_male.fit(X_train_male, y_train_male)
gb_model_female.fit(X_train_female, y_train_female)

joblib.dump(gb_model_male, 'gb_model_male.pkl')
joblib.dump(gb_model_female, 'gb_model_female.pkl')



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