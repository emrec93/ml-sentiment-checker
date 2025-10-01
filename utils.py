from typing import List
import pandas as pd


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
   
