import nltk
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

# Токенізація
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

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
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

# Збереження стоп-слів у файл
def save_stopwords_to_file(file_path):
    stop_words = set(stopwords.words('english'))
    with open(file_path, 'w', encoding='utf-8') as file:
        for word in stop_words:
            file.write(word + '\n')

# Обробка тексту
def process_text(file_path):
    text = read_rtf_file(file_path)
    tokens = tokenize(text)
    stemmed_tokens = stem_tokens(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens)
    filtered_tokens = remove_stopwords(lemmatized_tokens)
    return filtered_tokens

# Файл .rtf
file_path = 'large_text1.rtf'
processed_tokens = process_text(file_path)

# Вивід токенів
print(processed_tokens)

# Збереження результатів у файл
df = pd.DataFrame(processed_tokens, columns=["Token"])
df.to_csv('processed_tokens1.csv', index=False)

# Збереження стоп-слів у файл
stopwords_file_path = 'stopwords_list1.txt'
save_stopwords_to_file(stopwords_file_path)
print(f"Стоп-слова були збережені у {stopwords_file_path}")