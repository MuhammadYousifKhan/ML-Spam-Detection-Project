import joblib
import re
import string
import gradio as gr

# Load model, vectorizer, and threshold
model = joblib.load("C:/Users/Ammir Khan Khushik/Desktop/Internship Intellience/Tasks/Task1/SpamDetectionApp/spam_model.pkl")
vectorizer = joblib.load("C:/Users/Ammir Khan Khushik/Desktop/Internship Intellience/Tasks/Task1/SpamDetectionApp/tfidf_vectorizer.pkl")
threshold = joblib.load("C:/Users/Ammir Khan Khushik/Desktop/Internship Intellience/Tasks/Task1/SpamDetectionApp/best_threshold.pkl")


# Function to clean the input text (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Function to predict spam or ham
def predict_spam(message):
    cleaned = clean_text(message)
    vectorized = vectorizer.transform([cleaned])
    prob = model.predict_proba(vectorized)[:, 1][0]

    if prob >= threshold:
        return f"🔴 SPAM ({prob:.2f} probability)"
    else:
        return f"🟢 HAM ({prob:.2f} probability)"

# Gradio interface
interface = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=3, placeholder="Enter your message here..."),
    outputs="text",
    title="Spam Message Detector",
    description="Paste a message and detect if it's spam or not using a Logistic Regression model."
)

# Launch the app
interface.launch()
