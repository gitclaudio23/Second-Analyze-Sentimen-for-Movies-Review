import streamlit as st
import nltk
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download dataset
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews

# Load dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

data = pd.DataFrame({
    "text": [" ".join(words) for words, label in documents],
    "label": [label for _, label in documents]
})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorization
count_vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train SVM
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_count, y_train)

# Train Logistic Regression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)

# Streamlit App
st.title("🎬 Sentiment Analysis of Movie Reviews")
st.write("Masukkan ulasan film untuk dianalisis sentimennya!")

review_text = st.text_area("Masukkan teks ulasan:")
if st.button("Analisis Sentimen"):
    if review_text:
        review_count = count_vectorizer.transform([review_text])
        review_tfidf = tfidf_vectorizer.transform([review_text])
        
        pred_svm = svm_model.predict(review_count)[0]
        pred_logistic = logistic_model.predict(review_tfidf)[0]
        
        st.write("### Hasil Prediksi:")
        st.write(f"**SVM Model:** {pred_svm}")
        st.write(f"**Logistic Regression Model:** {pred_logistic}")
    else:
        st.warning("Harap masukkan teks ulasan!")

# Visualization of Word Cloud
def plot_wordcloud(texts, title):
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(" ".join(texts))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt)

if st.checkbox("Tampilkan Word Cloud"):
    plot_wordcloud(data['text'], "Word Cloud of Movie Reviews")

# Confusion Matrix Visualization
if st.checkbox("Tampilkan Confusion Matrix"):
    y_pred_svm = svm_model.predict(X_test_count)
    y_pred_logistic = logistic_model.predict(X_test_tfidf)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred_svm), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title("Confusion Matrix - SVM")
    
    sns.heatmap(confusion_matrix(y_test, y_pred_logistic), annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title("Confusion Matrix - Logistic Regression")
    
    st.pyplot(fig)

st.write("**Dibuat dengan ❤️ menggunakan Streamlit**")
