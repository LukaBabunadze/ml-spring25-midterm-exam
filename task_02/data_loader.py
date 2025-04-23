"""
Utility module for loading data
"""
import pandas as pd


def load_data(path: str):
    """
    Load dataset from CSV and return features and target.

    Parameters:
    - path: str, path to CSV file

    Returns:
    - X: pandas.DataFrame, feature columns
    - y: pandas.Series, target column
    """
    df = pd.read_csv(path)
    X = df[['words', 'links', 'capital_words', 'spam_word_count']]
    y = df['is_spam']
    return X, y