import streamlit as st
import pickle
import re

# Load the trained vectorizer & model
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Debugging prints (to verify correct feature count)
print("📍 Model expects features:", model.n_features_in_)
print("📍 Vectorizer vocabulary size:", len(tfidf_vectorizer.get_feature_names_out()))

# Ensure text preprocessing is identical to training
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to preprocess input and predict
def predict_message(msg):
    msg = preprocess_text(msg)  # ✅ Apply the same preprocessing as training
    msg_tfidf = tfidf_vectorizer.transform([msg])  # ✅ Use ONLY transform()

    # Debugging print (Check if transformed features match model input size)
    print("📢 Input message transformed features:", msg_tfidf.shape[1])

    prediction = model.predict(msg_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"

# Streamlit UI
st.title("Spam-Ham Classifier")
message = st.text_area("Enter a message:")
if st.button("Predict"):
    result = predict_message(message)
    st.write(f"Prediction: {result}")
