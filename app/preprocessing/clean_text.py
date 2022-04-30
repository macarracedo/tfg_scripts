import nltk
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from cleantext import clean

nltk.download('stopwords')

def remove_url(tweet):
    return re.sub(r'http\S+', '', tweet)

def preprocessing_pipeline(texts, cleantext=True, lemmatization=True, stemming=True, stpwrds=True):

    stop_words = stopwords.words('english')

    texts = np.array(texts)

    remove_line_breaks = np.vectorize(lambda text: text.replace("\n", " "))
    remove_puncts = np.vectorize(lambda text: text.replace(",", " ").replace(".", " ").replace("|", " "))
    remove_urls = np.vectorize(lambda text: remove_url(text))
    clean_text = np.vectorize(lambda text: clean(text,
                                                 fix_unicode=True,  # fix various unicode errors
                                                 to_ascii=True,  # transliterate to closest ASCII representation
                                                 lower=True,  # lowercase text
                                                 no_line_breaks=True,
                                                 no_urls=True,  # replace all URLs with a special token
                                                 no_emails=True,  # replace all email addresses with a special token
                                                 no_phone_numbers=True,
                                                 no_numbers=True,  # replace all numbers with a special token
                                                 no_digits=True,  # replace all digits with a special token
                                                 no_currency_symbols=True,
                                                 no_punct=True,  # remove punctuations
                                                 replace_with_punct="",
                                                 replace_with_url="",
                                                 replace_with_email="",
                                                 replace_with_phone_number="",
                                                 replace_with_digit="",
                                                 replace_with_number="",
                                                 replace_with_currency_symbol="",
                                                 lang="en"
                                                 ))

    remove_stopwords = np.vectorize(lambda text: ' '.join([word for word in text.split() if word not in stop_words]))
    word_lemmatize = np.vectorize(lambda text: ' '.join([wl.lemmatize(word) for word in text.split()]))
    word_stemming = np.vectorize(lambda text: ' '.join([ps.stem(word) for word in text.split()]))
    
    if cleantext:
        texts = remove_line_breaks(texts)
        texts = remove_puncts(texts)
        texts = remove_urls(texts)
        texts = clean_text(texts)

    if stpwrds:
        texts = remove_stopwords(texts)

    if lemmatization:
        wl = WordNetLemmatizer()
        texts = word_lemmatize(texts)

    if stemming:
        ps = PorterStemmer()
        texts = word_stemming(texts)


    return texts.tolist()