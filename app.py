import streamlit as st
import pickle
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to download a file from a URL
def download_file(url):
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    return BytesIO(response.content)

# URLs of the raw pickle files
model_url = 'https://github.com/Praveena-RP/Disaster-Tweet-Prediction/raw/main/svm_model.pkl'
vectorizer_url = 'https://github.com/Praveena-RP/Disaster-Tweet-Prediction/raw/main/tfidf_vectorizer.pkl'

# Download the model and vectorizer files
model_file = download_file(model_url)
vectorizer_file = download_file(vectorizer_url)

# Load the saved GridSearchCV model and TF-IDF vectorizer from the downloaded files
with model_file as f:
    grid_search_model = pickle.load(f)

with vectorizer_file as f:
    loaded_vectorizer = pickle.load(f)

# Extract the best model (SVM) from the GridSearchCV object
svm_model = grid_search_model.best_estimator_

# Title of the app
st.title("Disaster Text Prediction")

# Instructions for the user
st.write("Enter a text related to a natural disaster and get the predicted category.")

# Text input box for user input
user_input = st.text_area("Enter Text:", "Type a disaster-related event here...")

# Button to predict
if st.button("Predict"):
    if user_input:
        # Vectorize the input text using the loaded TF-IDF vectorizer
        user_input_tfidf = loaded_vectorizer.transform([user_input])

        # Predict the label using the loaded SVM model
        prediction = svm_model.predict(user_input_tfidf)

        # Display the input text and prediction with color formatting
        prediction_label = prediction[0]

        # Color formatting for prediction text
        st.markdown(f"### Your Input Text:")
        st.write(f"**{user_input}**")  # User input text in default color

        # Display prediction with different color
        st.markdown(f"### Predicted Label: <span style='color:red;'> {prediction_label} </span>", unsafe_allow_html=True)
    else:
        st.write("Please enter some text to get a prediction.")
