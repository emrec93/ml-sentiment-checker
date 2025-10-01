
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# 1. Load data
df = pd.read_csv("data/sentiment.csv")

# 2. Split into features (X) and labels (y)
X = df["text"]
y = df["label"]

# 3. Create pipeline (vectorizer + classifier)
#    The vectorizer turns text into numeric features and classifier uses those features to learn patterns and make predictions
#    make_pipeline chains them together into one model we can .fit() and .predict() with.
model = make_pipeline(CountVectorizer(), MultinomialNB())

# 4. Train
model.fit(X, y)

# 5. Save model
joblib.dump(model, "models/sentiment_model.pkl")

print("✅ Model trained and saved!")