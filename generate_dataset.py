import pandas as pd
import random

# Base positive and negative phrases
positive_phrases = [
    "Absolutely loved it!",
    "This film was amazing, so fun to watch!",
    "I can’t stop smiling after this movie",
    "Brilliant acting and perfect story!",
    "Totally recommend it to everyone",
    "Heartwarming and truly inspiring",
    "A masterpiece of filmmaking",
    "So funny and entertaining",
    "Exceeded all my expectations",
    "Clever and witty dialogue",
]

negative_phrases = [
    "Terrible movie, wasted my time",
    "Horrible acting, plot made no sense",
    "I didn’t enjoy it at all",
    "Boring and predictable story",
    "One of the worst films I’ve seen",
    "Painfully slow and long",
    "Completely forgettable",
    "Plot holes everywhere",
    "Disappointing from start to finish",
    "Awful pacing and weak acting",
]

# Emoji options
positive_emojis = ["😍", "🥰", "😁", "🌟", "💖", "🎶", "🥳", "😊"]
negative_emojis = ["😡", "😒", "😴", "😩", "😤", "😭", "💀", "😕"]

data = []

for _ in range(500):
    # Positive example
    text = random.choice(positive_phrases)
    if random.random() < 0.3:  # ~30% chance to add an emoji
        text += " " + random.choice(positive_emojis)
    data.append([text, "positive"])
    
    # Negative example
    text = random.choice(negative_phrases)
    if random.random() < 0.3:  # ~30% chance to add an emoji
        text += " " + random.choice(negative_emojis)
    data.append([text, "negative"])

# Create DataFrame
df = pd.DataFrame(data, columns=["text", "label"])

# Save to CSV
df.to_csv("data/sentiment_large_real.csv", index=False)
print("✅ Large dataset with emojis and variations created!")
