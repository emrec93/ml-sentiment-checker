from typing import List
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    """
    Clean text for preprocessing:
    - Lowercase
    - Remove punctuation (except emojis)
    - Remove stopwords
    - Remove extra whitespace
    """
    text = text.lower()
    # Keep letters, numbers, spaces, and emojis
    text = re.sub(r"[^a-z0-9\s\U0001F600-\U0001F64F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Remove stopwords
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text



def count_words(text: str) -> int:
    """
    Count the number of words in a given string.
    
    Args:
        text (str): The input sentence.

    Returns:
        int: Number of words in the sentence.
    """
    return len(text.split())


def is_prime(n: int) -> bool:
    """
    Check if a number is prime.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if prime, False otherwise.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True



def load_csv_summary(path: str) -> None:
    """
    Load a CSV file and print a quick summary.

    Args:
        path (str): Path to the CSV file.

    Returns:
        None
    """
    try:
        df = pd.read_csv(path)
        rows, cols = df.shape
        print(f"Dataset has {rows} rows and {cols} columns.")
        print(df.head())
        print(df["age"].mean())
    except:
        print("Error - something went wrong!")
   
