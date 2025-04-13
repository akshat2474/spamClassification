import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv('spam.csv', encoding='latin-1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    data['label'] = data['label'].map({'spam': 1, 'ham': 0})
    return data

# Train the model
@st.cache_resource
def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.3, random_state=42)
    vectorizer = CountVectorizer(stop_words='english')
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    
    model = MultinomialNB()
    model.fit(X_train_transformed, y_train)
    
    y_pred = model.predict(X_test_transformed)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "vectorizer": vectorizer,
        "model": model,
    }
    
    return metrics

# Visualization of confusion matrix
def plot_confusion_matrix(conf_matrix):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

# Set custom styles using Markdown and CSS
def set_custom_styles():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #ece9e6, #ffffff);
            font-family: Arial, sans-serif;
            color: #333333;
        }
        .title {
            text-align: center;
            font-size: 48px;
            font-weight: bold;
            color: #4E79A7;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            font-size: 24px;
            color: #555555;
            margin-bottom: 30px;
        }
        .metrics-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Streamlit App Layout
set_custom_styles()
st.markdown('<h1 class="title">Spam Email Classification App</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Classify emails as Spam or Ham using a machine learning model</p>', unsafe_allow_html=True)

data = load_data()
metrics = train_model(data)

# Sidebar options
menu = ["Test Email", "Metrics"]
choice = st.sidebar.radio("Menu", menu)

if choice == "Test Email":
    st.subheader("Test Your Email")
    
    # Two-column layout for better spacing
    col1, col2 = st.columns([3, 2])
    
    with col1:
        user_input = st.text_area("Enter the email content below:", height=200)
        
        if st.button("Classify"):
            vectorizer = metrics["vectorizer"]
            model = metrics["model"]
            
            user_input_transformed = vectorizer.transform([user_input])
            prediction = model.predict(user_input_transformed)[0]
            
            if prediction == 1:
                st.markdown('<div class="metrics-card" style="color:#E15759;">This email is classified as: SPAM</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="metrics-card" style="color:#59A14F;">This email is classified as: HAM</div>', unsafe_allow_html=True)
    
elif choice == "Metrics":
    st.subheader("Model Performance Metrics")
    
    # Display Metrics Cards in Columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metrics-card">Accuracy:<br><span style="color:#4E79A7;">{:.2f}</span></div>'.format(metrics["accuracy"]), unsafe_allow_html=True)
    
    with col2:
        precision_spam = metrics["classification_report"].split()[5]
        recall_spam = metrics["classification_report"].split()[6]
        
        st.markdown('<div class="metrics-card">Precision (Spam):<br><span style="color:#F28E2B;">{}</span></div>'.format(precision_spam), unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metrics-card">Recall (Spam):<br><span style="color:#E15759;">{}</span></div>'.format(recall_spam), unsafe_allow_html=True)
    
    # Confusion Matrix Visualization
    plot_confusion_matrix(metrics["confusion_matrix"])

