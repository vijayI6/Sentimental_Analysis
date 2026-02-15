# 🧠 Sentiment Analysis using Multinomial Naive Bayes

An end-to-end Natural Language Processing (NLP) project that classifies text into Positive or Negative sentiment using Machine Learning.

---

## 📌Project Overview

This project implements a complete Sentiment Analysis pipeline using **Multinomial Naive Bayes**, a probabilistic machine learning algorithm well-suited for text classification tasks.

The system processes raw textual data, converts it into numerical features using vectorization techniques, and trains a classification model to predict sentiment polarity.

---

## 🚀 Features

- Text preprocessing pipeline
- Feature extraction using TF-IDF / Bag of Words
- Multinomial Naive Bayes classifier
- Model evaluation using multiple performance metrics
- Sentiment prediction for custom user input

---

## 🏗️ Project Workflow

Raw Text  
→ Text Cleaning & Preprocessing  
→ Feature Extraction (TF-IDF / CountVectorizer)  
→ Train-Test Split  
→ Model Training (MultinomialNB)  
→ Evaluation  
→ Prediction  

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **NLTK**
- **Scikit-learn**
- **Jupyter Notebook**

---

## 📂 Project Structure

Sentiment_Analysis_final.ipynb  → Main notebook  
README.md                       → Project documentation  

---

## ⚙️ Implementation Details

### 1. Data Preprocessing

- Convert text to lowercase  
- Remove punctuation & special characters  
- Remove stopwords  
- Tokenization  
- Stemming / Lemmatization  

This step removes noise and prepares clean input for model training.

---

### 2. Feature Engineering

Text data is converted into numerical format using:

- **Bag of Words (CountVectorizer)**
- **TF-IDF (Term Frequency – Inverse Document Frequency)**

TF-IDF improves model performance by reducing the weight of common words.

---

### 3. Model Used

**Multinomial Naive Bayes (MultinomialNB)**

This probabilistic classifier is based on Bayes’ Theorem and works efficiently for text classification problems involving word frequency features.

---

### 4. Model Evaluation
The model is evaluated using the following metrics:

#### 🔹 Accuracy Score
Measures the percentage of correct predictions out of total predictions.

Accuracy = Correct Predictions / Total Predictions  

Example:  
If accuracy = 0.80 → The model correctly predicts 80% of the data.

---

#### 🔹 Confusion Matrix

A Confusion Matrix shows detailed classification results:

- True Positive (TP)  
- True Negative (TN)  
- False Positive (FP)  
- False Negative (FN)  

It helps identify where the model makes mistakes.

---

#### 🔹 Precision

Precision = TP / (TP + FP)  

Measures how many predicted positive values are actually positive.  
High precision means fewer false positives.

---

#### 🔹 Recall

Recall = TP / (TP + FN)  

Measures how many actual positive values were correctly predicted.  
High recall means fewer missed positive cases.

---

#### 🔹 F1-Score

F1 = 2 × (Precision × Recall) / (Precision + Recall)  

F1-score balances both Precision and Recall.  
It is especially useful when the dataset is imbalanced.

---

## 🧪 Example Prediction

### Enter a review:  The product good

### The review: 'The product good' is predicted Postive Review
---