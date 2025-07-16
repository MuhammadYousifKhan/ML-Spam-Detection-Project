# 📧 ML Spam Detection Project

A Machine Learning-based web app that detects whether a message is **Spam** or **Ham** using **Logistic Regression** and **TF-IDF Vectorization**.  
Built and deployed as part of my internship at **Internship Intelligence**.

---

## 🚀 Features

- 🔍 Logistic Regression classifier
- ✉️ Classifies text messages as spam or ham
- 📊 TF-IDF vectorization with unigrams and bigrams
- 🎯 Custom probability threshold (0.3) for better recall
- 🌐 Gradio web app interface (runs locally)
- 🧪 Jupyter notebook showing full ML pipeline

---

## 📊 Model Performance

| Metric     | Score      |
|------------|------------|
| Accuracy   | 96.32%     |
| Precision  | 97.39%     |
| Recall     | 74.67%     |
| F1 Score   | 84.53%     |

---

## 📁 Project Structure


---

## 🛠️ Tech Stack

- Python 3.x
- Scikit-learn
- Pandas
- Gradio
- Joblib
- Regular Expressions

---

## 📊 Dataset

The dataset used is [`spam.csv`](./spam.csv), sourced from the  
[UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).  
It contains 5,000+ labeled SMS messages categorized as "spam" or "ham".

---

## 🧠 How It Works

1. **Text Preprocessing:** Lowercasing, removing digits and punctuation  
2. **Vectorization:** TF-IDF with (1,2)-grams  
3. **Model Training:** Logistic Regression  
4. **Threshold Tuning:** Using `0.3` instead of default `0.5` to improve recall  
5. **Deployment:** Using Gradio to create a local web UI for prediction

---

## ▶️ How to Run

```bash
# Step 1: Clone the repo
git clone https://github.com/MuhammadYousifKhan/ML-Spam-Detection-Project.git
cd ML-Spam-Detection-Project

# Step 2: Install required packages
pip install gradio scikit-learn joblib pandas

# Step 3: Run the app
python app.py
