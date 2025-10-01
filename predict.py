import joblib

# Load trained model
model = joblib.load("models/sentiment_large_model.pkl")

# Example sentences
examples = [
    "What a wonderful film!",
    "This was the worst experience ever.",
    "I really loved the story and acting.",
    "Boring and disappointing movie.",
    "It was a masterclass in how to do something horribly wrong!"
]

for text in examples:
    prediction = model.predict([text])[0]
    print(f"{text} -> {prediction}")
