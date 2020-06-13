import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
nltk.download('stopwords')
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()
stop_words = list(stopwords.words('english'))

def string_normalization(s, lemmatizer=None, stop_words=None):
    """
    s: pandas.core.series.Series
    """
    sn = s.str.lower()
    sn = sn.str.replace(r'[\n|\r]',' ')
    sn = sn.str.replace(r'(?<=\d)[,](?=\d)','') # 12,000 ==> 12000
    sn = sn.str.replace(r'(\d{1,2})([a-z\-\*#])', r'\1 \2') # 12.0oz ==> 12.0 oz
    sn = sn.str.replace(r'([a-z\-\*#])(\d{1,2})', r'\1 \2') # pack12 ==> pack 12
    sn = sn.str.replace(r'[\?:!;\|&\+\*\"\%,#\(\)\[\]\{\}/\'\"]',' ')
    sn = sn.str.replace(r'(?<![a-z])\-(?![a-z])',' ')
    sn = sn.str.replace(r"\'s",'')
    if stop_words:
        stop_word_regex = "\\b(" + "|".join(stop_words) + ")\\b"
        sn = sn.str.replace(stop_word_regex, '')
    if lemmatizer:
        sn = sn.apply(lambda s : ' '.join([lemmatizer.lemmatize(word, pos="v") for word in str(s).split()]))
        sn = sn.apply(lambda s : ' '.join([lemmatizer.lemmatize(word) for word in str(s).split()]))
    sn = sn.replace('\s+', ' ')
    return sn