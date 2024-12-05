import sqlite3
from chatbot.config import DATABASE_PATH

def connect_to_db():
    """Connect to the SQLite database and return the connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    return conn

def insert_health_advice(topic, advice_text):
    """Insert new health advice into the database."""
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO HealthAdvice (topic, advice_text) VALUES (?, ?)", (topic, advice_text))

    conn.commit()
    conn.close()

def fetch_health_advice(topic):
    """Fetch health advice for a specific topic from the database."""
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT advice_text FROM HealthAdvice WHERE topic = ?", (topic,))
    advice = cursor.fetchone()
    conn.close()

    if advice:
        return advice[0]
    else:
        return None

def predict_and_add_to_db(user_input, model_path):
    """Predict with the model and add the prediction to the database."""
    model_prediction = "Here's a prediction based on your health query."
    topic = user_input.lower().strip()
    insert_health_advice(topic, model_prediction)
    return model_prediction
