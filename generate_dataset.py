import pandas as pd
import random

positive_phrases = [
    "Absolutely loved it! 😍",
    "This film was amazing, so fun to watch!",
    "I can’t stop smiling after this movie 😄",
    "Brilliant acting and perfect story!",
    "Totally recommend it to everyone 👍",
]

negative_phrases = [
    "Terrible movie, wasted my time 😡",
    "Horrible acting, plot made no sense",
    "I didn’t enjoy it at all 👎",
    "Boring and predictable story",
    "One of the worst films I’ve seen 😠",
]

data = []
for _ in range(500):
    data.append([random.choice(positive_phrases), "positive"])
    data.append([random.choice(negative_phrases), "negative"])

df = pd.DataFrame(data, columns=["text", "label"])
df.to_csv("data/sentiment_large_real.csv", index=False)
print("✅ Large dataset with emojis and variations created!")
