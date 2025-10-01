# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
from utils import clean_text

# Load trained model
model = joblib.load("models/sentiment_large_model.pkl")

app = FastAPI(title="Sentiment Analysis API")

# Request body for single sentence
class Sentence(BaseModel):
    text: str

# Request body for batch sentences
class SentencesBatch(BaseModel):
    texts: List[str]

@app.get("/")
def read_root():
    return {"message": "Sentiment Analysis API is running!"}

@app.post("/predict")
def predict_sentiment(sentence: Sentence):
    cleaned = clean_text(sentence.text)
    pred = model.predict([cleaned])[0]
    confidence = max(model.predict_proba([cleaned])[0])
    return {
        "text": sentence.text,
        "sentiment": pred,
        "confidence": confidence
    }

@app.post("/predict_batch")
def predict_batch(batch: SentencesBatch):
    results = []
    for text in batch.texts:
        cleaned = clean_text(text)
        pred = model.predict([cleaned])[0]
        confidence = max(model.predict_proba([cleaned])[0])
        results.append({
            "text": text,
            "sentiment": pred,
            "confidence": confidence
        })
    return results
