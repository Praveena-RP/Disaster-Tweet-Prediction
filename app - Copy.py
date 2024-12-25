import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved GridSearchCV model and TF-IDF vectorizer from pickle files
model_path = 'D:/Disaster_mp1/svm_model.pkl'
vectorizer_path = 'D:/Disaster_mp1/tfidf_vectorizer.pkl'

with open(model_path, 'rb') as model_file:
    grid_search_model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    loaded_vectorizer = pickle.load(vectorizer_file)

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
