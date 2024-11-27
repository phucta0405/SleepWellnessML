import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load datasets
data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
data1 = pd.read_csv("student_sleep_patterns.csv")

# Rename columns for consistency
data.rename(columns={"Sleep Duration": "Sleep_Duration"}, inplace=True)

# Filter relevant columns
data_filtered = data[["Age", "Sleep_Duration", "Gender"]]
data1_filtered = data1[["Age", "Sleep_Duration", "Gender"]]

# Combine datasets
combined_data = pd.concat([data_filtered, data1_filtered], ignore_index=True)

# Drop rows with missing values
combined_data.dropna(inplace=True)

# Create dummy variables for Gender
combined_data = pd.get_dummies(combined_data, columns=["Gender"], drop_first=True)

# Prepare features (X) and target (y)
X = combined_data[["Age", "Gender_Male"]]
y = combined_data["Sleep_Duration"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a single Gradient Boosting Regressor model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)

# Predict based on user input
try:
    user_age = float(input("Input your age: "))
    gender_input = input("Input your gender (Male/Female): ").strip()

    if gender_input not in ["Male", "Female"]:
        print("Invalid gender input. Please input 'Male' or 'Female'.")
    else:
        # Encode gender as a dummy variable
        gender_male = 1 if gender_input == "Male" else 0
        predicted_sleep_duration = gb_model.predict([[user_age, gender_male]])[0]
        print(f"Predicted Sleep Duration for Age {user_age} ({gender_input}): {predicted_sleep_duration:.2f} hours")
except ValueError:
    print("Invalid age input. Please input a numeric value.")
