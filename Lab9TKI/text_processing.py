import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Ініціалізація лематизатора
lemmatizer = WordNetLemmatizer()

# Список стоп-слів
stop_words = set(
    stopwords.words('english'))  # Ви можете змінити на 'russian' для російської або додати свої українські стоп-слова


# Функція для попередньої обробки тексту
def preprocess_text(text):
    # Токенізація тексту
    tokens = word_tokenize(text)
    print("Токенізація:", tokens)

    # Лематизація та видалення стоп-слів
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.lower() not in stop_words and token.isalnum()]
    print("Лематизація та видалення стоп-слів:", tokens)

    return ' '.join(tokens)


# Папка з текстовими файлами
folder_path = 'Corpus'

# Зчитування та обробка вмісту всіх текстових файлів у папці
documents = []
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
            text = file.read()
            print(f"\n--- Документ: {filename} ---")
            print("Оригінальний текст:", text)

            processed_text = preprocess_text(text)
            print("Оброблений текст:", processed_text)

            documents.append(processed_text)

# Ініціалізація TfidfVectorizer
vectorizer = TfidfVectorizer()

# Підгонка та перетворення корпусу
tfidf_matrix = vectorizer.fit_transform(documents)

# Отримання словника (всі унікальні слова)
feature_names = vectorizer.get_feature_names_out()

# Виведення TF-IDF для кожного слова в кожному документі
tfidf_data = tfidf_matrix.toarray()

# Формування ТОП-3 ключових слів для кожного документа
top_n = 3
for i, doc in enumerate(documents):
    print(f"\nДокумент {i + 1}:")
    # Отримання індексів ТОП-3 слів
    top_indices = np.argsort(tfidf_data[i])[-top_n:]
    top_words = [feature_names[index] for index in top_indices]
    print("ТОП-3 ключових слова:", ", ".join(top_words))
    print()
