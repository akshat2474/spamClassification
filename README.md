# 📧 Spam Detection using Naive Bayes

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success)
![Made with Scikit-learn](https://img.shields.io/badge/Made%20with-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)

A simple and effective spam detection model built using **Multinomial Naive Bayes** and the **SMS Spam Collection Dataset**. This project demonstrates how to classify messages into `spam` or `ham` using Natural Language Processing and Machine Learning.

---

## 🔗 Try It Online

**Check out the interactive Streamlit app here:**  
[https://akshat2474-spamclassification-app-r0jdr9.streamlit.app/](https://akshat2474-spamclassification-app-r0jdr9.streamlit.app/)

---

## 📂 Dataset

> [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)  
A collection of 5,574 SMS messages in English, labeled as either **ham** (legitimate) or **spam** (unwanted promotional content).

---

## 🔧 Tech Stack

-  Python 3.8+
-  Pandas
-  Scikit-learn
-  CountVectorizer (Bag of Words model)
-  Naive Bayes Classifier
-  Streamlit (for web interface)

---

## 🚀 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/akshat2474/spamClassification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📈 How It Works

- Load and clean the dataset
- Convert labels (`ham`, `spam`) to binary (`0`, `1`)
- Vectorize messages using `CountVectorizer` (removing stopwords)
- Train a `MultinomialNB` classifier
- Evaluate the model using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
- **Classify custom messages live via Streamlit**

---

## 🧪 Sample Prediction (Code-based)

```python
new_email = ["Win a $1000 gift card now! Click here."]
new_email_transformed = vectorizer.transform(new_email)
prediction = model.predict(new_email_transformed)
print("Prediction (0=ham, 1=spam):", prediction[0])
```

---

## 📊 Model Evaluation

- **Accuracy:** 98.32% 
- **Confusion Matrix**: Breakdown of True/False Positives & Negatives
- **Precision/Recall**: Useful for spam-heavy datasets
- Visualized via bar charts in the Streamlit interface

---

## 🖥️ Streamlit App Features

-  Beautiful UI with custom CSS styling
-  Test your own email/SMS messages in real time
-  Visualize accuracy, precision, recall, and F1-score
-  View the first few rows of the dataset
-  Efficient caching for faster load times

---

## 👤 Author

**Akshat Singh**  
[LinkedIn](https://www.linkedin.com/in/akshat-singh-48a03b312/) • [GitHub](https://github.com/akshat2474)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

