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
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text

# Function to output prediction label
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

# Streamlit interface
st.title("Fake News Detection")

# User input for news text
news_text = st.text_area("Enter News Text", "Type your news article here...")

# Prediction button for Logistic Regression model
if st.button("Predict"):
    if news_text:
        # Clean the input news text
        cleaned_text = wordopt(news_text)

        # Vectorize the cleaned text
        new_x_test = [cleaned_text]
        new_xv_test = vectorizer.transform(new_x_test)

        # Get prediction from the Logistic Regression model
        pred_LR = LR.predict(new_xv_test)

        # Show prediction
        st.subheader("Prediction from Logistic Regression:")
        st.write(f"Logistic Regression Prediction: {output_label(pred_LR[0])}")
    else:
        st.error("Please enter some news text for prediction.")
