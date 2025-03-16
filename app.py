import streamlit as st
import joblib
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Load only the Logistic Regression model
LR = joblib.load(r'logistic_regression_model.pkl')

# Load the vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Ensure you save this as well, or recreate it if needed

# Function to clean text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Function to output prediction label
def output_label(n):
    return "Not A Fake News" if n == 1 else "Fake News"

# Streamlit interface with custom CSS
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        html, body, [class*="st"] {
            font-family: 'Poppins', sans-serif;
        }

        .stApp {
            background: url('https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8dGVjaG5vbG9neXxlbnwwfHwwfHx8MA%3D%3D') no-repeat center center fixed;
            background-size: cover;
        }
        
        .main-title {
            text-align: center;
            font-size: 150px;
            font-weight: 600;
            color: #ff4500;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .stTextArea textarea {
            background-color: rgba(0, 0, 0, 0.8) !important;
            border-radius: 10px;
        }

        .stButton>button {
            background-color: black;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            transition: 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #ff1c1c;
        }

        .prediction {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            color: white;
            padding: 10px;
            background: rgba(0,0,0,0.7);
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<p class="main-title">📰 Fake News Detection 📰</p>', unsafe_allow_html=True)

# User input for news text
news_text = st.text_area("Enter News Text", "Type your news article here...")

# Prediction button for Logistic Regression model
if st.button("🔍 Predict"):
    if news_text:
        # Clean the input news text
        cleaned_text = wordopt(news_text)

        # Vectorize the cleaned text
        new_x_test = [cleaned_text]
        new_xv_test = vectorizer.transform(new_x_test)

        # Get prediction from the Logistic Regression model
        pred_LR = LR.predict(new_xv_test)

        # Show prediction
        st.markdown(f'<div class="prediction">Prediction: {output_label(pred_LR[0])}</div>', unsafe_allow_html=True)
    else:
        st.error("⚠️ Please enter some news text for prediction.")
