import requests

url = "http://127.0.0.1:5000/predict-sleep-quality"
data = {
    "age": 25,
    "gender": "Male",
    "sleep_duration": 7.5,
    "study_hours": 3,
    "screen_time": 4,
    "caffeine_intake": 2,
    "physical_activity": 45
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())