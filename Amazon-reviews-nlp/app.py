
import streamlit as st
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# --- 1. Load Model and Vectorizer ---
model_path = 'logistic_regression_count_model.joblib'
vectorizer_path = 'count_vectorizer.joblib'

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    st.success("Model and Vectorizer loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model or vectorizer file not found. Please ensure '{model_path}' and '{vectorizer_path}' exist.")
    st.stop()

# --- 2. Text Cleaning Setup ---
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

download_nltk_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Fixed syntax error below (removed duplicate 'char')
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ''.join([char for char in text if not char.isdigit()])
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- 3. Load Dataset for Data Overview ---
try:
    df = pd.read_csv(r'C:\Users\Hasan Zaman\Downloads\ITSimplera Solutions Internship\Week 5 (NLP)\amazon_reviews.csv')
    df['verified_reviews'] = df['verified_reviews'].fillna('')
    df['cleaned_reviews'] = df['verified_reviews'].apply(clean_text)
    st.success("Dataset loaded successfully for Data Overview.")
except Exception:
    df = None

# --- Streamlit App Layout ---
st.set_page_config(page_title="Amazon Review Sentiment Analysis", layout="centered")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Sentiment Predictor"])

# --- Page: Home ---
if page == "Home":
    st.title("Welcome to the Amazon Review Sentiment Analyzer")
    st.write("This application allows us to analyze the sentiment of Amazon product reviews. "
             "We can explore the dataset, and predict the sentiment (positive or negative) of new review texts.")
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?auto=format&fit=crop&w=1000&q=80", use_container_width=True, caption="Sentiment Analysis")
    st.markdown("### About This Project")
    st.write("This project was developed as part of a machine learning assignment, "
             "focusing on Natural Language Processing (NLP) techniques to classify "
             "customer feedback. We used various text vectorization methods and "
             "classification models to identify the most effective approach.")

# --- Page: Data Overview ---
elif page == "Data Overview":
    st.title("Dataset Overview")
    st.write("Here we provide a quick overview of the Amazon Reviews dataset that we used for training our models.")

    if df is not None:
        st.markdown("### Raw Data Sample")
        st.dataframe(df[['verified_reviews', 'feedback', 'rating']].head())

        st.markdown("### Dataset Information")
        st.write(f"Total reviews: {len(df)}")
        st.write(f"Columns: {df.columns.tolist()}")

        st.markdown("### Sentiment Distribution")
        feedback_counts = df['feedback'].value_counts().rename(index={1: 'Positive', 0: 'Negative'})
        st.bar_chart(feedback_counts)
        st.write("The distribution of feedback (sentiment):")
        st.dataframe(feedback_counts.to_frame())

        st.markdown("### Rating Distribution")
        rating_counts = df['rating'].value_counts().sort_index()
        st.bar_chart(rating_counts)
        st.write("The distribution of ratings:")
        st.dataframe(rating_counts.to_frame())
    else:
        st.info("Data overview is unavailable because the dataset could not be loaded.")

# --- Page: Sentiment Predictor ---
elif page == "Sentiment Predictor":
    st.title("Predict Review Sentiment")
    st.write("Enter an Amazon product review below to predict its sentiment (Positive or Negative).")

    user_input = st.text_area("Enter your review here:", "This product is absolutely amazing! I love it.")

    if st.button("Predict Sentiment"):
        if user_input:
            cleaned_input = clean_text(user_input)

            if cleaned_input:
                vectorized_input = vectorizer.transform([cleaned_input])

                prediction = model.predict(vectorized_input)
                prediction_proba = model.predict_proba(vectorized_input)

                sentiment = "Positive" if prediction[0] == 1 else "Negative"
                confidence = prediction_proba[0][prediction[0]] * 100

                st.write("\n--- Prediction ---")
                color = "green" if sentiment == "Positive" else "red"
                st.markdown(f"**Predicted Sentiment:** <span style='color: {color}; font-size: 20px;'>{sentiment}</span>", unsafe_allow_html=True)
                st.write(f"**Confidence:** {confidence:.2f}%")

                st.markdown("\n### Original vs. Cleaned Input")
                st.write(f"**Original Review:** {user_input}")
                st.write(f"**Cleaned Review:** {cleaned_input}")
            else:
                st.warning("The input review resulted in empty text after cleaning. Please try a different review.")
        else:
            st.warning("Please enter a review to predict its sentiment.")
