# app.py
import streamlit as st
import pickle
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")

# Load vectorizer
cv = pickle.load(open("cv.pkl", "rb"))


models = {
    "Logistic Regression": pickle.load(open("LogisticRegressionmodel.pkl", "rb")),
    "Decision Tree": pickle.load(open("DecisionTreemodel.pkl", "rb")),
    "Random 8Forest": pickle.load(open("RandomForestmodel.pkl", "rb")),
    "Naive Bayes": pickle.load(open("NaiveBayesmodel.pkl", "rb")),
    "KNN": pickle.load(open("KNNmodel.pkl", "rb"))
}

st.title("Restaurant Review Sentiment Analysis")
st.write("Select a model and enter a review to predict sentiment as **Positive** or **Negative**.")

# Model selection
model_name = st.selectbox("Choose Model", list(models.keys()))
model = models[model_name]

# Text input
user_input = st.text_area("Your Review", "")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        ps = PorterStemmer()
        review = re.sub("[^A-Za-z]", " ", user_input)
        review = review.lower()
        review = review.split()
        all_stopwords = stopwords.words("english")
        if "not" in all_stopwords:
            all_stopwords.remove("not")
        review = [ps.stem(word) for word in review if word not in all_stopwords]
        review = " ".join(review)
        review_vector = cv.transform([review]).toarray()

        pred = model.predict(review_vector)[0]

        if pred == 1:
            st.success(f" Positive Review ")
        else:
            st.error(f" Negative Review ")
