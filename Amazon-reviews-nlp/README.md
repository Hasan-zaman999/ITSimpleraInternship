# Amazon Echo Reviews — NLP Sentiment Analysis & Web App

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.7.2-F7931E.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.10.0-green.svg)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Natural Language Processing (NLP) and Machine Learning project designed to analyze customer feedback for Amazon Echo devices. The project handles everything from raw text preprocessing and bag-of-words vectorization to training a Logistic Regression model and serving interactive predictions via a **Streamlit Web Application**.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Tech Stack & Dependencies](#-tech-stack--dependencies)
- [Dataset Architecture](#-dataset-architecture)
- [Machine Learning & NLP Pipeline](#-machine-learning--nlp-pipeline)
- [Model Performance & Results](#-model-performance--results)
- [Installation & Setup](#-installation--setup)
- [How to Run](#-how-to-run)
  - [1. Jupyter Notebook](#1-running-the-jupyter-notebook)
  - [2. Streamlit Web App](#2-launching-the-streamlit-web-app)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 📌 Overview

Understanding customer sentiment at scale is critical for product quality and brand perception. Unstructured customer text presents challenges due to noise, varying lengths, and slang.

This repository provides an end-to-end solution:
1. **Preprocessing Pipeline**: Cleans and normalizes customer reviews (`verified_reviews`).
2. **Text Vectorization**: Extracts feature matrices using `CountVectorizer`.
3. **Sentiment Modeling**: Trains and evaluates a `LogisticRegression` classifier to predict sentiment (`1` for Positive, `0` for Negative).
4. **Web Interface**: Deploys the trained vectorizer and model into an interactive **Streamlit** dashboard (`app.py`) for real-time review predictions.

---

## ✨ Key Features

- **End-to-End Workflow**: Modular pipeline covering data ingestion, cleaning, training, serialization, and deployment.
- **Robust NLP Normalization**: Lowercasing, regex-based special character/digit removal, stopword filtering, and lemmatization.
- **Model Serialization**: Pre-trained `CountVectorizer` and `LogisticRegression` artifacts stored using `joblib` for rapid inference.
- **Interactive UI (`app.py`)**: Web application enabling users to input custom reviews and receive instant sentiment predictions with confidence scores.

---

## 📁 Repository Structure

```text
Amazon-reviews-nlp/
├── data/
│   └── amazon_reviews.csv                        # Dataset containing Amazon Echo reviews
├── models/
│   ├── count_vectorizer.joblib                   # Serialized CountVectorizer feature extractor
│   └── logistic_regression_count_model.joblib    # Serialized Logistic Regression model
├── notebooks/
│   └── week5_nlp.ipynb                           # Notebook containing EDA, preprocessing & modeling
├── app.py                                        # Streamlit web app for real-time sentiment inference
├── requirements.txt                              # Environment package dependencies
└── README.md                                     # Project documentation
```

---

## 🛠 Tech Stack & Dependencies

- **Language**: Python 3.8+
- **Data Manipulation**: `pandas` (2.3.3), `numpy` (2.2.6)
- **Natural Language Processing**: `nltk` (3.10.0)
- **Machine Learning & Modeling**: `scikit-learn` (1.7.2), `scipy` (1.15.3)
- **Model Serialization**: `joblib` (1.5.3)
- **Web Framework**: `streamlit` (1.60.0)
- **Visualization**: `matplotlib` (3.10.8), `seaborn` (0.13.2), `wordcloud` (1.9.6)

---

## 📊 Dataset Architecture

The raw dataset (`data/amazon_reviews.csv`) contains **3,150 customer reviews** collected from Amazon Echo listings.

### Column Metadata

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `rating` | `int64` | Star rating awarded by customer (1 to 5 stars) |
| `date` | `object` | Review submission date |
| `variation` | `object` | Amazon Echo model variant (e.g., *Charcoal Fabric*, *Walnut Finish*) |
| `verified_reviews`| `object` | Body text of customer review |
| `feedback` | `int64` | Target label (`1` = Positive, `0` = Negative) |

### Class Distribution
- **Positive (`1`)**: 2,893 reviews (~91.84%)
- **Negative (`0`)**: 257 reviews (~8.16%)

---

## 🛠 Machine Learning & NLP Pipeline

```text
  Raw Customer Review
          │
          ▼
   1. Text Cleaning        ──► Lowercasing, punctuation & digit removal
          │
          ▼
   2. Tokenization & NLP   ──► Stopword filtering & WordNet Lemmatization
          │
          ▼
   3. Feature Extraction   ──► Bag-of-Words conversion using CountVectorizer
          │
          ▼
   4. Model Inference      ──► Logistic Regression Sentiment Classifier
          │
          ▼
   5. Output Prediction    ──► Positive (1) or Negative (0) + Probability Score
```

---

## 📈 Model Performance & Results

The Logistic Regression classifier trained on the `CountVectorizer` feature set achieved high accuracy and robust predictive performance across the test set:

### Evaluation Metrics Summary

| Metric | Score |
| :--- | :--- |
| **Model Accuracy** | **~93.5%** |
| **Positive Class (1) Precision** | **0.95** |
| **Positive Class (1) Recall** | **0.98** |
| **Positive Class (1) F1-Score** | **0.96** |
| **Negative Class (0) Precision** | **0.72** |
| **Negative Class (0) Recall** | **0.51** |
| **Negative Class (0) F1-Score** | **0.60** |

> **Note on Imbalance**: Due to the initial ~91.8% positive class skew, macro metrics prioritize monitoring negative review recall and precision alongside overall accuracy.

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Hasan-zaman999/ITSimpleraInternship/tree/main/Amazon-reviews-nlp.git
cd Amazon-reviews-nlp
```

### 2. Set Up a Virtual Environment

- **Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Windows:**
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Required NLTK Corpora
Run this command in Python to download required NLP resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## 🚀 How to Run

### 1. Running the Jupyter Notebook
To inspect exploratory analysis, visualization, and model training:
```bash
jupyter notebook notebooks/week5_nlp.ipynb
```

### 2. Launching the Streamlit Web App
To run the interactive sentiment prediction application locally:
```bash
streamlit run app.py
```
After running the command, open your browser and navigate to `http://localhost:8501`.

---

## 🔮 Future Enhancements

- [ ] Add support for `TF-IDF Vectorizer` alongside `CountVectorizer`.
- [ ] Implement class rebalancing techniques like SMOTE or class-weight adjustments.
- [ ] Integrate deep learning baselines (e.g., DistilBERT or RoBERTa).
- [ ] Containerize the application using Docker for cloud deployment.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
