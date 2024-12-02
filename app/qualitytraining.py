import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

def load_data():
    data1 = pd.read_csv("../data/student_sleep_patterns.csv")
    data2 = pd.read_csv("../data/Sleep_Efficiency.csv")

    data2["Bedtime"] = pd.to_datetime(data2["Bedtime"])
    data2["Wakeup time"] = pd.to_datetime(data2["Wakeup time"])
    data2["Sleep_Duration"] = (data2["Wakeup time"] - data2["Bedtime"]).dt.total_seconds() / 3600
    data2["Sleep_Duration"] = data2["Sleep_Duration"].apply(lambda x: x + 24 if x < 0 else x)

    data2['Exercise frequency'] = data2['Exercise frequency'].replace({
        1.0: 15, 2.0: 20, 3.0: 35, 4.0: 60, 5.0: 75
    }).infer_objects(copy=False)

    data2['Sleep efficiency'] = data2['Sleep efficiency'] * 10

    data2.rename(columns={
        "Exercise frequency": "Physical_Activity",
        "Sleep efficiency": "Sleep_Quality",
        "Caffeine consumption": "Caffeine_Intake"
    }, inplace=True)

    columns_to_keep = ["Age", "Gender", "Sleep_Duration", 
                       "Caffeine_Intake", "Physical_Activity", "Sleep_Quality"]
    data1 = data1[columns_to_keep]
    data2 = data2[columns_to_keep]
    data1 = data1.reset_index(drop=True)
    data2 = data2.reset_index(drop=True)

    combined_data = pd.concat([data1, data2], ignore_index=True)

    return combined_data

def preprocess_data(data):
    data = data[data['Gender'].isin(['Male', 'Female'])]
    data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
    X = data[["Age", "Gender", "Sleep_Duration", 
              "Caffeine_Intake", "Physical_Activity"]]
    Y = data[["Sleep_Quality"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "scaler.pkl")

    return X_scaled, Y, scaler


def train_model(X_train, y_train):
    rf_model = RandomForestRegressor(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_dist,
        n_iter=100, 
        cv=3, 
        n_jobs=-1, 
        verbose=2, 
        scoring='neg_mean_squared_error',
        random_state=42
    )

    random_search.fit(X_train, y_train.values.ravel())

    best_model = random_search.best_estimator_
    print(f"Best Parameters: {random_search.best_params_}")

    return best_model

def predict_optimal_sleep(model, scaler, age, gender, real_duration, caffeine_intake, physical_activity):
    sleep_durations = np.round(np.linspace(4, 12, 81), decimals=1)
    max_quality = -np.inf
    best_duration = 0

    features = np.array([[age, gender, real_duration, caffeine_intake, physical_activity]])
    scaled_features = scaler.transform(features)
    real_quality = model.predict(scaled_features)[0]

    duration_quality = []

    for sleep_duration in sleep_durations:
        features = np.array([[age, gender, sleep_duration, caffeine_intake, physical_activity]])
        scaled_features = scaler.transform(features)

        predicted_quality = model.predict(scaled_features)[0]

        duration_quality.append((sleep_duration, predicted_quality))

        if predicted_quality > max_quality:
            max_quality = predicted_quality
            best_duration = sleep_duration
        
    durations, qualities = zip(*duration_quality)

    return real_quality, best_duration, max_quality, np.array(durations), np.array(qualities)

if __name__ == "__main__":
    data = load_data()
    X, Y, scaler = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    joblib.dump(model, "optimized_rf_model.pkl")

    try:
        user_age = float(input("Input your age: "))
        gender_input = input("Input your gender (Male/Female): ").strip()
        caffeine_intake = float(input("Input your daily Caffeine Intake (cups): "))
        physical_activity = float(input("Input your level of Physical Activity (minutes/day): "))

        if gender_input not in ["Male", "Female"]:
            print("Invalid gender input. Please input 'Male' or 'Female'.")
        else:
            gender_encoded = 1 if gender_input == "Male" else 0

            print("hello")

            # Predict optimal sleep duration
            real_quality, best_duration, max_quality, _, _ = predict_optimal_sleep(
                model, scaler, user_age, 7, gender_encoded, caffeine_intake, physical_activity
            )

            print(f"Optimal Sleep Duration: {best_duration:.2f} hours")
            print(f"Maximum Predicted Sleep Quality: {max_quality:.2f}")
    except ValueError:
        print("Invalid input. Please ensure all inputs are in the correct format.")
