import os
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from striprtf.striprtf import rtf_to_text

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Читання файлу .rtf
def read_rtf_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        rtf_content = file.read()
    text = rtf_to_text(rtf_content)
    return text

# Читання текстового файлу
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Токенізація
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

# Видалення пунктуації
def remove_punctuation(tokens):
    table = str.maketrans('', '', string.punctuation)
    stripped_tokens = [token.translate(table) for token in tokens if token.translate(table)]
    return stripped_tokens

# Стемінг
def stem_tokens(tokens):
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return stemmed_tokens

# Лематизація
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Видалення стоп-слів
def remove_stopwords(tokens, custom_stopwords):
    filtered_tokens = [token for token in tokens if token.lower() not in custom_stopwords]
    return filtered_tokens

# Завантаження власного списку стоп-слів з файлу
def load_custom_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        custom_stopwords = set(file.read().split())
    return custom_stopwords

# Обробка тексту
def process_text(file_path, custom_stopwords):
    if file_path.endswith('.rtf'):
        text = read_rtf_file(file_path)
    else:
        text = read_text_file(file_path)
    tokens = tokenize(text)
    tokens = remove_punctuation(tokens)
    stemmed_tokens = stem_tokens(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens)
    filtered_tokens = remove_stopwords(lemmatized_tokens, custom_stopwords)
    return filtered_tokens

# Обробка корпусу текстових файлів
def process_corpus(corpus_dir, stopwords_file_path):
    custom_stopwords = load_custom_stopwords(stopwords_file_path)
    all_processed_tokens = []

    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
            if file.endswith('.rtf') or file.endswith('.txt'):
                file_path = os.path.join(root, file)
                processed_tokens = process_text(file_path, custom_stopwords)
                all_processed_tokens.extend(processed_tokens)

    return all_processed_tokens

# Шлях до корпусу текстових файлів і файлу зі стоп-словами
corpus_dir = r'Corpus'
stopwords_file_path = r'stopwords_ua.txt'

# Обробка корпусу
processed_tokens = process_corpus(corpus_dir, stopwords_file_path)

# Вивід токенів
print(processed_tokens)

# Збереження результатів у файл
df = pd.DataFrame(processed_tokens, columns=["Token"])
df.to_csv('processed_corpus_tokens.csv', index=False)