import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
data1 = pd.read_csv("student_sleep_patterns.csv")

# Select only Age and Sleep Duration columns
data_filtered = data[["Age", "Sleep Duration"]]
data1_filtered = data1[["Age", "Sleep Duration"]]

# Concatenate datasets
combined_data = pd.concat([data_filtered, data1_filtered], ignore_index=True)

# Drop rows with missing values
combined_data.dropna(inplace=True)

# Feature (Age) and Target (Sleep Duration)
X = combined_data[["Age"]]
y = combined_data["Sleep Duration"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


gb_model = GradientBoostingRegressor(random_state=42)


gb_model.fit(X_train, y_train)


y_pred = gb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test data:", mse)

user_age = input("Input your age: ") 
predicted_sleep_duration = gb_model.predict([[user_age]])[0]
print(f"Predicted Sleep Duration for Age {user_age}: {predicted_sleep_duration:.2f} hours")
