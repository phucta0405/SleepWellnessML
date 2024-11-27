import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Feature selection: Age as X and Sleep Duration as y
X = data[["Age"]]  # Features (Age)
y = data["Sleep Duration"]  # Target (Sleep Duration)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(random_state=42)

# Fit the model on training data
gb_model.fit(X_train, y_train)

# Make predictions on test data
y_pred = gb_model.predict(X_test)

# Evaluate the model: Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test data:", mse)

# Predict sleep duration for a new age input
user_age = 30  # Replace with any age input
predicted_sleep_duration = gb_model.predict([[user_age]])[0]
print(f"Predicted Sleep Duration for Age {user_age}: {predicted_sleep_duration:.2f} hours")
