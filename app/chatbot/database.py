import sqlite3
import config

def fetch_health_advice(topic):
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT advice_text FROM HealthAdvice WHERE topic = ?", (topic,))
    advice = cursor.fetchone()

    conn.close()

    if advice:
        return advice[0]
    else:
        return "Sorry, I don't have advice on that topic right now."

def insert_health_advice(topic, advice_text):
    conn = sqlite3.connect(config.DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("INSERT INTO HealthAdvice (topic, advice_text) VALUES (?, ?)", (topic, advice_text))

    conn.commit()
    conn.close()
