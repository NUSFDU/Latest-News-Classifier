import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

WEIGHT_UNITS = ['eg', 'gg', 'mg', 'pg', 'tg', 'yg', 'zg', 'ag', 'cg', 'dag', 'dg', 'fg', 'g', 'hg', 'kg', 'lb', 'long_ton', 'mg', 'ng', 'oz', 'pg', 'short_ton', 'stone', 'tonne', 'ug', 'yg', 'zg', 'attogram', 'centigram', 'decagram', 'decigram', 'exagram', 'femtogram', 'gigagram', 'gram', 'hectogram', 'kilogram', 'long ton', 'mcg', 'megagram', 'metric ton', 'metric tonne', 'microgram', 'milligram', 'nanogram', 'ounce', 'petagram', 'picogram', 'pound', 'short ton', 'teragram', 'ton', 'yoctogram', 'yottagram', 'zeptogram', 'zetagram']
VOLUME_UNITS = ['al', 'attoliter', 'attolitre', 'centiliter', 'centilitre', 'cl', 'cubic centimeter', 'cubic foot', 'cubic inch', 'cubic meter', 'cubic_centimeter', 'cubic_foot', 'cubic_inch', 'cubic_meter', 'cup', 'dal', 'decaliter', 'decalitre', 'deciliter', 'decilitre', 'dl', 'el', 'exaliter', 'exalitre', 'femtoliter', 'femtolitre', 'fl', 'fl ounce', 'fl oz', 'fl\\.ounce', 'fl\\.oz', 'fluid', 'fluid oz', 'gallon', 'gigaliter', 'gigalitre', 'gl', 'hectoliter', 'hectolitre', 'hl', 'imperial pint', 'imperial quart', 'imperial tablespoon', 'imperial teaspoon', 'imperial_g', 'imperial_oz', 'imperial_pint', 'imperial_qt', 'imperial_tbsp', 'imperial_tsp', 'kiloliter', 'kilolitre', 'kl', 'l', 'liter', 'litre', 'megaliter', 'megalitre', 'microliter', 'microlitre', 'milliliter', 'millilitre', 'ml', 'nanoliter', 'nanolitre', 'nl', 'petaliter', 'petalitre', 'picoliter', 'picolitre', 'pint', 'pl', 'qt', 'quart', 'table spoon', 'tablespoon', 'tbsp', 'tea spoon', 'teaspoon', 'teraliter', 'teralitre', 'tl', 'tsp', 'ul', 'us cup', 'us fluid ounce', 'us gallon', 'us ounce', 'us pint', 'us quart', 'us tablespoon', 'us teaspoon', 'us_cup', 'us_pint', 'us_qt', 'us_tbsp', 'us_tsp', 'yl', 'yoctoliter', 'yoctolitre', 'yottaliter', 'yottalitre', 'zeptoliter', 'zeptolitre', 'zetaliter', 'zetalitre', 'zl']
COUNTS_UNITS = ['(?<=\d )can', 'bag', 'bottle', 'box', 'bundle', 'canister', 'canistre', 'count', 'dozen', 'dz', 'jar', 'kit', 'lot', 'pack', 'pc', 'piece', 'pill', 'score', 'set', 'sheet', 'shot', 'tablet', 'tube', 'unit']


def extract_key_token(s, token_list, verbose=False):
    """
    s: pandas.core.series.Series
    token_list: list(str)
    returns: pandas.core.series.Series(list)
    """
    token_regex = "\\b(" + "|".join(token_list) + ")[s\.]*\\b"
    if verbose:
        print(token_regex)
    sn = s.str.lower()
    sn = sn.str.findall(token_regex)
    return sn
