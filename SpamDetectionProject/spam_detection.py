# ==============================
# SMS SPAM DETECTION PROJECT
# FINAL STABLE VERSION
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download stopwords once
nltk.download('stopwords')

# Load stopwords into memory (important fix)
stop_words = set(stopwords.words('english'))

# ---------- Load Dataset ----------
print("Loading dataset...")
data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

print("Dataset Loaded Successfully!")
print(data.head())

# ---------- Convert Labels ----------
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# ---------- Text Cleaning ----------
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

print("Cleaning text...")
data['clean_msg'] = data['message'].apply(clean_text)

# ---------- Feature Extraction ----------
print("Converting text into numerical features...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['clean_msg'])
y = data['label_num']

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ---------- Train ----------
print("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model training completed!")

# ---------- Prediction ----------
y_pred = model.predict(X_test)

# ---------- Evaluation ----------
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------- Confusion Matrix ----------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i][j], ha="center", va="center")

plt.tight_layout()
plt.show()

# ---------- Custom Prediction ----------
def predict_spam(text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    result = model.predict(vector)
    
    if result[0] == 1:
        return "Spam Message"
    else:
        return "Not Spam"

print("\nTesting custom messages:")
print("Example 1:", predict_spam("Congratulations! You won a free lottery"))
print("Example 2:", predict_spam("Hey, are we meeting tomorrow?"))

# ---------- Save Model ----------
pickle.dump(model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and Vectorizer saved successfully!")
print("\nProject Completed Successfully!")