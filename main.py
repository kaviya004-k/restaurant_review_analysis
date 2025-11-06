# train_model.py
import numpy as np
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download("stopwords")

# Load dataset
dataset = pd.read_csv("restaurant\Restaurant_Reviews.tsv", delimiter="\t")

# Text preprocessing
corpus = []
ps = PorterStemmer()
all_stopwords = stopwords.words("english")
if "not" in all_stopwords:
    all_stopwords.remove("not")

for i in range(0, len(dataset)):
    review = re.sub("[^A-Za-z]", " ", dataset["Review"][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in all_stopwords]
    review = " ".join(review)
    corpus.append(review)

# Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.20, random_state=0)

# Models dictionary
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=0),
    "DecisionTree": DecisionTreeClassifier(random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=0),
    "NaiveBayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    acc = accuracy_score(ytest, ypred)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(ytest, ypred))
    print("Classification Report:\n", classification_report(ytest, ypred))
    results[name] = acc

# Save all models and vectorizer
for name, model in models.items():
    with open(f"{name}model.pkl", "wb") as f:
        pickle.dump(model, f)

with open("cv.pkl", "wb") as f:
    pickle.dump(cv, f)

for name, acc in results.items():
    print(f"{name}: {acc:.4f}")
