# BBC News Category Classification with NLP

## Project Overview

This project aims to classify BBC news articles into predefined categories using Natural Language Processing (NLP) techniques and Machine Learning algorithms.

The dataset contains news articles from five categories:

- Business
- Entertainment
- Politics
- Sport
- Technology

Text preprocessing, stopword removal, TF-IDF vectorization, and machine learning models were applied to automatically predict the category of a news article.

---

## Dataset Information

- Dataset: BBC News Dataset
- Number of Articles: 2,225
- Number of Categories: 5
- Features:
  - Title
  - Content
- Target:
  - Category

---

## Data Analysis

Several exploratory data analysis (EDA) steps were performed:

- Dataset structure examination
- Missing value analysis
- Category distribution visualization
- Text length analysis
- Average word count analysis
- Most frequent words analysis
- WordCloud visualization

---

## Text Preprocessing

The following NLP preprocessing techniques were applied:

- Lowercase conversion
- Special character removal
- Number removal
- Stopword removal
- Text cleaning using Regular Expressions

---

## Feature Engineering

TF-IDF (Term Frequency – Inverse Document Frequency) was used to transform text documents into numerical vectors.

```python
TF-IDF Features: 5000
```

Resulting matrix size:

```python
(2225, 5000)
```

---

## Machine Learning Models

Two machine learning algorithms were trained and compared:

### 1. Multinomial Naive Bayes

Suitable for text classification problems and commonly used with TF-IDF features.

### 2. Logistic Regression

A linear classification algorithm that performs extremely well in NLP classification tasks.

---

## Model Results

| Model | Accuracy |
|---------|---------:|
| Multinomial Naive Bayes | 97.08% |
| Logistic Regression | 97.30% |

The Logistic Regression model achieved the highest performance and was selected as the final model.

---

## Streamlit Application

A Streamlit web application was developed where users can enter a news article and instantly receive a predicted category.

The application:

- Accepts user news text
- Performs preprocessing
- Applies TF-IDF transformation
- Predicts category using the trained Logistic Regression model
- Displays prediction results

### Live Demo

[Live Demo](https://huggingface.co/spaces/MSK34/bbc-news-category-classification-with-nlp)

---

## Project Structure

```text
bbc-news-category-classification
│
├── src
│   ├── streamlit_app.py
│   ├── news_category_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── BBC_News_Category_Classification.ipynb
├── requirements.txt
└── README.md
```

---

## Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Streamlit
- WordCloud

---

## Conclusion

In this project, BBC news articles were successfully classified into five categories using NLP techniques and machine learning algorithms. After preprocessing and TF-IDF vectorization, Logistic Regression achieved the best performance with approximately 97.30% accuracy. The results demonstrate that traditional NLP approaches combined with machine learning can provide highly accurate solutions for automatic news categorization tasks.

---
