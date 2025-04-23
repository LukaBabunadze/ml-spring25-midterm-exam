"""
Command-line tool to classify a raw email text as spam or not spam.
"""
import argparse
import joblib
import re
import numpy as np
from data_loader import load_data  # for column order only


def extract_features(text: str):
    # words count
    words = len(text.split())
    # links count
    links = len(re.findall(r'https?://', text))
    # capital words count
    caps = sum(1 for w in text.split() if w.isupper())
    # spam word count (example list)
    spam_words = ['free', 'win', 'winner', 'cash', 'prize']
    spam_count = sum(text.lower().count(sw) for sw in spam_words)
    return np.array([[words, links, caps, spam_count]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Email text to classify')
    args = parser.parse_args()

    model = joblib.load('task_02/logistic_model.joblib')
    features = extract_features(args.text)
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0, pred]

    label = 'Spam' if pred == 1 else 'Not Spam'
    print(f"Prediction: {label} (probability: {prob:.2f})")