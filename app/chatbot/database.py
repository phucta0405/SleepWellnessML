import sqlite3
import config

def connect_to_db():
    """Connect to the SQLite database and return the connection."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    return conn

def insert_health_advice(topic, advice_text):
    """Insert new health advice into the database."""
    conn = connect_to_db()
    cursor = conn.cursor()

    # Insert the health advice into the HealthAdvice table
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
    # Make a prediction using the model (this is a placeholder for actual prediction logic)
    # Let's assume the prediction is a health advice related to the input
    model_prediction = "Here's a prediction based on your health query."

    # Extract a topic from user_input (e.g., a keyword for the advice)
    topic = user_input.lower().strip()

    # Insert the model's prediction into the database
    insert_health_advice(topic, model_prediction)

    return model_prediction
