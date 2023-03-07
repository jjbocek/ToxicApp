#Text Preprocessing
import re # for number removal
import string # for punctutation removal

import nltk
## for stopword removal
from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords= stopwords.words('english')
## lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer= WordNetLemmatizer()
from nltk.tokenize import word_tokenize

import pickle  #save variables to file


# Text Preprocessing

# Removing punctuations'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
def remove_punc(comment):
    nonPunc = "".join([letter for letter in comment if letter not in string.punctuation])
    return nonPunc


# Lowering the text
def toLower(comment):
    return comment.lower()


# Removing numbers
def replace_numbers(comment):
    """Replace all interger occurrences in
    list of tokenized words with textual representation"""
    return re.sub(r'\d+', '', comment)


# Removing whitespaces
def remove_space(comment):
    return comment.strip()


# Tokenization
def text2word(comment):
    return word_tokenize(comment)


# Removing Stop words
def remove_stopW(words, stopWords):
    return [word for word in words if word not in stopWords]


# Lemmatization
def lematizer(words):
    lemmatizer = WordNetLemmatizer()
    lemm_comm = [lemmatizer.lemmatize(word) for word in words]
    return lemm_comm


def lematizer_verb(words):
    lemmatizer = WordNetLemmatizer()
    lemm_comm = [lemmatizer.lemmatize(word, "v") for word in words]
    return lemm_comm


def clean_comment(comment):
    comment = remove_punc(comment)
    comment = toLower(comment)
    comment = replace_numbers(comment)
    comment = remove_space(comment)
    words = text2word(comment)
    words = remove_stopW(words, stopWords)
    words = lematizer(words)
    words = lematizer_verb(words)

    return ' '.join(words)