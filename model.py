import pandas as pd
import re
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ─── Load Dataset ───────────────────────────────────────────────────────────
data = pd.read_excel("ai_vs_human_text-1.xlsx")
print(f"✅ Dataset loaded: {len(data)} samples")
print(f"   Label distribution: {data['label'].value_counts().to_dict()}")

# ─── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['clean_text'] = data['text'].apply(preprocess)

# ─── Features & Labels ──────────────────────────────────────────────────────
X = data['clean_text']
y = data['label']

# ─── Vectorizer (Unigrams + Bigrams for richer features) ────────────────────
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=5000)
X_vec = vectorizer.fit_transform(X)

# ─── Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ─── Model 1: Logistic Regression ───────────────────────────────────────────
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"\n📊 Logistic Regression Accuracy: {lr_acc * 100:.2f}%")
print(classification_report(y_test, lr_preds))

# ─── Model 2: Naive Bayes ───────────────────────────────────────────────────
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_preds)
print(f"📊 Naive Bayes Accuracy: {nb_acc * 100:.2f}%")
print(classification_report(y_test, nb_preds))

# ─── Save Models ────────────────────────────────────────────────────────────
pickle.dump(lr_model, open("lr_model.pkl", "wb"))
pickle.dump(nb_model, open("nb_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("✅ All models saved: lr_model.pkl, nb_model.pkl, vectorizer.pkl")
