import streamlit as st
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Define the wordopt function
def wordopt(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d', '', text)
        text = re.sub(r'\n', ' ', text)
    else:
        text = ''
    return text

@st.cache_resource
def load_model_and_vectorizer(filename):
    with open(filename, 'rb') as file:
        model_and_vectorizer = pickle.load(file)
    return model_and_vectorizer


def load_data():
    fake_data = pd.read_csv('Fake.csv', on_bad_lines='skip', encoding='utf-8', engine='python')
    true_data = pd.read_csv('True.csv', on_bad_lines='skip', encoding='utf-8', engine='python')
    fake_data["class"] = 0
    true_data["class"] = 1
    news = pd.concat([fake_data, true_data], axis=0)
    news = news.drop(["title", "subject", "date"], axis=1)
    return news


# Initialize session state variables
if 'model_and_vectorizer' not in st.session_state:
    st.session_state.model_and_vectorizer = None

st.title("Fake News Detection System")

menu = ["Home", "Manual Testing", "Model Report"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Home - Fake News Detection")
    st.write("This application uses a pre-trained machine learning model to classify news articles as fake or genuine.")
    
    # Load pre-trained model and vectorizer
    filename = "fake_news_detection.sav"
    if st.session_state.model_and_vectorizer is None:
        st.text('Loading pre-trained model...')
        try:
            st.session_state.model_and_vectorizer = load_model_and_vectorizer(filename)
            st.text('Pre-trained model loaded successfully!')
            st.write(st.session_state.model_and_vectorizer)  # Debug print to check if model and vectorizer are loaded
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()  # Stop execution if model loading fails

elif choice == "Manual Testing":
    st.subheader("Manual Testing")
    news_article = st.text_area("Enter a news article for classification:", "")
    if st.button("Classify"):
        if st.session_state.model_and_vectorizer is None:
            st.error('Model not loaded. Please go to the "Home" section to load the pre-trained model.')
        else:
            LR, vectorizer = st.session_state.model_and_vectorizer
            # Preprocess the input news article
            news_article = wordopt(news_article)
            news_article = str(news_article)  # Ensure it's converted to string
            # Transform the input using the loaded vectorizer
            transformed_input = vectorizer.transform([news_article])
            # Predict using the loaded LR model
            prediction = LR.predict(transformed_input)
            result = "It is a fake news" if prediction[0] == 0 else "It is a genuine news"
            st.success(result)

elif choice == "Model Report":
    st.subheader("Model Report")
    if st.session_state.model_and_vectorizer:
        LR, vectorizer = st.session_state.model_and_vectorizer
        # Assuming load_data is defined elsewhere in your script
        news = load_data()
        x = news['text']
        y = news['class']
        _, x_test, _, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        xv_test = vectorizer.transform(x_test)
        prediction_lr = LR.predict(xv_test)
        report = classification_report(y_test, prediction_lr, output_dict=True)
        st.write(report)
    else:
        st.error('Model report not available. Please go to the "Home" section to load the pre-trained model.')
