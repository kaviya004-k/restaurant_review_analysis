🍽️ Restaurant Reviews Sentiment Analysis  
This project applies Natural Language Processing (NLP) and Machine Learning techniques to analyze and classify restaurant reviews as positive or negative. It involves comprehensive text preprocessing — including tokenization, stopword removal, stemming, and vectorization using Bag of Words (BoW)
=======
🥗 Restaurant Reviews Sentiment Analysis

This project focuses on performing Sentiment Analysis on restaurant reviews to determine whether a review expresses a positive or negative opinion.
It combines Natural Language Processing (NLP) techniques with various Machine Learning models and a Streamlit web application for user interaction.

🚀 Project Overview

Cleaned and preprocessed textual data from restaurant reviews.

Converted text into numerical features using the Bag of Words (BoW) model.

Trained and compared five machine learning algorithms to find the best performer.

Built an interactive Streamlit app to predict sentiment from new user input.

Used pickle to save and load the trained model efficiently.

🧠 Models Used

Logistic Regression

Naïve Bayes

Support Vector Machine (SVM)

Decision Tree Classifier

Random Forest Classifier

Each model was evaluated using metrics like accuracy, precision, recall, and F1-score to determine the best performance for sentiment prediction.

🧰 Tech Stack

Python

Pandas, NumPy

NLTK (Natural Language Toolkit)

Scikit-learn

Streamlit

Pickle

📊 Dataset

The dataset used is Restaurant_Reviews.tsv, containing text reviews and their corresponding sentiment labels (positive / negative).

⚙️ Workflow

Import and explore dataset

Clean text data (tokenization, stopword removal, stemming)

Convert text into numerical features using CountVectorizer

Train and evaluate machine learning models

Save the best model using pickle

Deploy the model with a Streamlit app

🧩 Streamlit Usage

To run the web app:

streamlit run app.py


Make sure app.py, the trained model file (e.g., model.pkl), and the vectorizer file are in the same directory.






>>>>>>> a17a8e8 (Added restaurant review dataset and analysis code)
