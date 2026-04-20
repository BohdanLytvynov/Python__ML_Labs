import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import string

# Зразки текстів
texts = [
"Cats are beautiful animals.",
"Dogs are loyal and friendly.",
"Birds can fly and sing beautifully."
]

# Очищення тексту
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return " ".join(tokens)

cleaned_texts = [preprocess(text) for text in texts]
print(cleaned_texts)

#3. Побудова TF-IDF моделі
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
print("Слова:", vectorizer.get_feature_names_out())
print("TF-IDF матриця:\n", tfidf_matrix.toarray())

#4. Використання Word2Vec (Gensim)
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
# Підготовка токенів
tokenized = [word_tokenize(doc.lower()) for doc in texts]
# Навчання Word2Vec
model = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=1, workers=4)
# Приклад: отримати вектор слова
print("Vector for 'cats', model.wv['cats']")