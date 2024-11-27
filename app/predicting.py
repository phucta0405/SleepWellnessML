import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

datapath = "../data/Sleep_health_and_lifestyle_dataset.csv"
data1path = "../data/student_sleep_patterns.csv"
data = pd.read_csv(datapath)
data1 = pd.read_csv(data1path)

data.rename(columns={"Sleep Duration": "Sleep_Duration"}, inplace=True)

data_filtered = data[["Age", "Sleep_Duration", "Gender"]]
data1_filtered = data1[["Age", "Sleep_Duration", "Gender"]]

combined_data = pd.concat([data_filtered, data1_filtered], ignore_index=True)

combined_data.dropna(inplace=True)

male_data = combined_data[combined_data["Gender"] == "Male"]
female_data = combined_data[combined_data["Gender"] == "Female"]

X_male = male_data[["Age"]]
y_male = male_data["Sleep_Duration"]

X_female = female_data[["Age"]]
y_female = female_data["Sleep_Duration"]


X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(
    X_male, y_male, test_size=0.2, random_state=42
)
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(
    X_female, y_female, test_size=0.2, random_state=42
)

gb_model_male = GradientBoostingRegressor(random_state=42)
gb_model_female = GradientBoostingRegressor(random_state=42)

gb_model_male.fit(X_train_male, y_train_male)
gb_model_female.fit(X_train_female, y_train_female)

import joblib
joblib.dump(gb_model_male, 'gb_model_male.pkl')
joblib.dump(gb_model_female, 'gb_model_female.pkl')
