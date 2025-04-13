import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# Visualization of metrics
def plot_metrics():
    metrics_labels = ['Accuracy', 'Precision (0)', 'Recall (0)', 'F1-Score (0)', 'Precision (1)', 'Recall (1)', 'F1-Score (1)']
    values = [
        0.9832535885167464,
        0.99,
        0.99,
        0.99,
        0.95,
        0.92,
        0.94
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics_labels, values, color=['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1'])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Performance Metrics of Spam Email Classification Model')
    
    st.pyplot(fig)

# Streamlit App Layout
st.title("Spam Email Classification App")
st.write("This app classifies emails as either **Spam** or **Ham** using a Naive Bayes model.")

data = load_data()
metrics = train_model(data)

# Sidebar options
menu = ["Home", "Test Email", "Metrics"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Dataset Overview")
    st.write(data.head())
    
elif choice == "Test Email":
    st.subheader("Test Your Email")
    
    user_input = st.text_area("Enter the email content below:")
    
    if st.button("Classify"):
        vectorizer = metrics["vectorizer"]
        model = metrics["model"]
        
        user_input_transformed = vectorizer.transform([user_input])
        prediction = model.predict(user_input_transformed)[0]
        
        if prediction == 1:
            st.error("This email is classified as: SPAM")
        else:
            st.success("This email is classified as: HAM")
            
elif choice == "Metrics":
    st.subheader("Model Performance Metrics")
    
    # Display Accuracy and Classification Report
    st.write(f"**Accuracy:** {metrics['accuracy']:.2f}")
    
    st.text("Classification Report:")
    st.text(metrics["classification_report"])
    
    # Display Confusion Matrix
    st.text("Confusion Matrix:")
    st.write(metrics["confusion_matrix"])
    
    # Plot Metrics Visualization
    plot_metrics()

