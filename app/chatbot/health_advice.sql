
CREATE TABLE IF NOT EXISTS HealthAdvice (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    advice_text TEXT NOT NULL
);


INSERT INTO HealthAdvice (topic, advice_text)
VALUES
    ('sleep', 'To improve sleep, try maintaining a consistent bedtime, reducing screen time before bed, and keeping your room cool and dark.'),
    ('exercise', 'Exercise is beneficial for overall health. Aim for at least 30 minutes of moderate activity most days of the week.'),
    ('diet', 'Eat a balanced diet rich in vegetables, fruits, lean proteins, and whole grains for optimal health.');
