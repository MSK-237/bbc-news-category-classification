import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


st.set_page_config(
    page_title="BBC News Category Classification",
    page_icon="📰",
    layout="centered"
)

st.title("📰 BBC News Category Classification")

st.write(
    "Bu uygulama, haber metnini analiz ederek haberin hangi kategoriye ait olduğunu tahmin eder. "
    "Model, TF-IDF ve Logistic Regression kullanılarak eğitilmiştir."
)


@st.cache_resource
def load_model():
    model = joblib.load("news_category_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    return model, tfidf


model, tfidf = load_model()


extra_words = {
    "said", "mr", "mrs", "new", "year", "years",
    "people", "time", "world", "uk", "just",
    "make", "told", "best", "bn", "like"
}

stop_words = set(ENGLISH_STOP_WORDS).union(extra_words)


def temizle(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)

    words = [
        word
        for word in text.split()
        if word not in stop_words
        and len(word) > 2
    ]

    return " ".join(words)


st.subheader("Haber Metni Girin")

user_text = st.text_area(
    "Tahmin yapılacak haber metnini yazın:",
    height=180,
    placeholder="Example: The government announced a new economic policy today..."
)

if st.button("Kategori Tahmini Yap"):

    if user_text.strip() == "":
        st.warning("Lütfen bir haber metni girin.")

    else:
        clean_text = temizle(user_text)

        text_vector = tfidf.transform([clean_text])

        prediction = model.predict(text_vector)[0]

        st.success(f"Tahmin Edilen Kategori: {prediction}")

        st.write("Temizlenmiş Metin:")
        st.write(clean_text)