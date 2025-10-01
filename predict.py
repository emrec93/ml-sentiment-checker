import joblib

# Load trained model
model = joblib.load("models/sentiment_model.pkl")

# Example sentences
examples = [
    "What a wonderful film!",
    "This was the worst experience ever."
]

for text in examples:
    prediction = model.predict([text])[0]
    print(f"{text} -> {prediction}")
