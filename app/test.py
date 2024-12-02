import requests

url = "http://127.0.0.1:5000/chat"
data = {
    "message": "Hello?",
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())