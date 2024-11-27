import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load datasets
data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
data1 = pd.read_csv("student_sleep_patterns.csv")

# Standardize column names
data.rename(columns={"Sleep Duration": "Sleep_Duration"}, inplace=True)

# Select only relevant columns
data_filtered = data[["Age", "Sleep_Duration", "Gender"]]
data1_filtered = data1[["Age", "Sleep_Duration", "Gender"]]

# Concatenate datasets
combined_data = pd.concat([data_filtered, data1_filtered], ignore_index=True)

# Drop rows with missing values
combined_data.dropna(inplace=True)

# Separate genders
male_data = combined_data[combined_data["Gender"] == "Male"]
female_data = combined_data[combined_data["Gender"] == "Female"]

# Features and target for males
X_male = male_data[["Age"]]
y_male = male_data["Sleep_Duration"]

# Features and target for females
X_female = female_data[["Age"]]
y_female = female_data["Sleep_Duration"]

# Train-test split
X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(
    X_male, y_male, test_size=0.2, random_state=42
)
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(
    X_female, y_female, test_size=0.2, random_state=42
)

# Model training
gb_model_male = GradientBoostingRegressor(random_state=42)
gb_model_female = GradientBoostingRegressor(random_state=42)

gb_model_male.fit(X_train_male, y_train_male)
gb_model_female.fit(X_train_female, y_train_female)

# Evaluate models
y_pred_male = gb_model_male.predict(X_test_male)
mse_male = mean_squared_error(y_test_male, y_pred_male)
print("Male Mean Squared Error:", mse_male)

y_pred_female = gb_model_female.predict(X_test_female)
mse_female = mean_squared_error(y_test_female, y_pred_female)
print("Female Mean Squared Error:", mse_female)

# Get user input
try:
    user_age = float(input("Input your age: "))
    gender_input = input("Input your gender (Male/Female): ").strip()

    if gender_input not in ["Male", "Female"]:
        print("Invalid gender input. Please input 'Male' or 'Female'.")
    else:
        if gender_input == "Male":
            predicted_sleep_duration = gb_model_male.predict([[user_age]])[0]
        else:
            predicted_sleep_duration = gb_model_female.predict([[user_age]])[0]
        print(f"Predicted Sleep Duration for Age {user_age}: {predicted_sleep_duration:.2f} hours")
except ValueError:
    print("Invalid age input. Please input a numeric value.")
