import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_json('D:/Disaster_mp1/balanced_dataset.json')  # Make sure it's the correct format

# Example dataset structure assumption
# df['text'] contains the text and df['label'] contains the labels
X = df['text']
y = df['label']

# Split the dataset into training and testing sets (adjust test_size as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use TF-IDF to vectorize the text data
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Define the SVM model
svm_model = SVC()

# Parameter grid for GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Set up GridSearchCV with 3 splits (since we have 3 samples in this case)
grid_search = GridSearchCV(svm_model, param_grid, cv=3)

# Fit the grid search to the training data
grid_search.fit(X_train_tfidf, y_train)

# Print the best parameters and the best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate on the test set
y_pred = grid_search.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict labels for custom text samples
custom_texts = [
    "tsunami in srilanka",
    "hail storm in japan"
]

# Vectorize the custom texts using the same TF-IDF vectorizer
custom_texts_tfidf = vectorizer.transform(custom_texts)

# Predict labels for the custom texts
custom_predictions = grid_search.predict(custom_texts_tfidf)

# Print predictions for custom texts
for text, prediction in zip(custom_texts, custom_predictions):
    print(f"Text: '{text}' => Predicted Label: '{prediction}'")



import pickle

# Save the trained SVM model and TF-IDF vectorizer as pickle files in the specified directory
model_path = 'D:/Disaster_mp1/svm_model.pkl'
vectorizer_path = 'D:/Disaster_mp1/tfidf_vectorizer.pkl'

# Save the grid search (SVM model) as a pickle file
with open(model_path, 'wb') as model_file:
    pickle.dump(grid_search, model_file)

# Save the TF-IDF vectorizer as a pickle file
with open(vectorizer_path, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"Model and vectorizer saved as pickle files at {model_path} and {vectorizer_path}")

