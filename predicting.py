import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

X = data[["Age"]]  # Features (Age)
y = data["Sleep Duration"]  # Target (Sleep Duration)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


gb_model = GradientBoostingRegressor(random_state=42)


gb_model.fit(X_train, y_train)


y_pred = gb_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on test data:", mse)

user_age = input() 
predicted_sleep_duration = gb_model.predict([[user_age]])[0]
print(f"Predicted Sleep Duration for Age {user_age}: {predicted_sleep_duration:.2f} hours")
