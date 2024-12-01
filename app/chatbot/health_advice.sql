CREATE TABLE IF NOT EXISTS advice (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    user_query TEXT NOT NULL,
    advice_text TEXT NOT NULL,
);

INSERT INTO advice (category, user_query, advice_text, source)
VALUES 
    ("Sleep", "can't sleep", "Try creating a bedtime routine, avoid caffeine before bed, and ensure your bedroom is dark and quiet.")
    ("Diet", "healthy eating", "Include a variety of fruits, vegetables, whole grains, and lean proteins in your daily meals."),
    ("Stress", "anxious", "Practice deep breathing exercises, mindfulness, or yoga to help manage stress levels.");


SELECT * FROM advice;
