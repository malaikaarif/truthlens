import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading dataset...")
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

true_df["label"] = 1  # real
fake_df["label"] = 0  # fake

df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df["text"] = df["title"] + " " + df["text"]
df = df[["text", "label"]].dropna()

print(f"Total articles: {len(df)}")
print(f"Real: {df['label'].sum()} | Fake: {len(df) - df['label'].sum()}")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModel saved as model.pkl")