import emoji as emj # pip install emoji==1.7
from tqdm import tqdm
tqdm.pandas()
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
EMOJIS = emj.UNICODE_EMOJI["en"]
import pandas as pd
pd.options.mode.chained_assignment = None


def apply_stemming(tokenized_column):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokenized_column]

def extract_emoji(text):
    list = []
    for emoji in EMOJIS:
        if emoji in text:
            list.append(emoji)
    return list

def removetextuseless(my_string):
    x = re.sub(r'http\S+', '', my_string)
    x = re.sub(r'www\S+', '', x)
    x = re.sub("@[A-Za-z0-9_]+", "", x)
    x = re.sub("#[A-Za-z0-9_]+", "", x)
    return re.sub(r'[^\w\s]|_', '', x)

def rejoin_words(tokenized_column):
    return (" ".join(tokenized_column))

def tokenize(column):
    tokens = nltk.word_tokenize(column)
    return [w for w in tokens if w.isalpha()]

def remove_stopwords(tokenized_column, stops):
    return [word for word in tokenized_column if not word in stops]

def preprocess_Text(text):
    stops = set(stopwords.words("english"))
    emoji = extract_emoji(text)
    text = removetextuseless(text)
    text = tokenize(text)
    text = remove_stopwords(text, stops)
    text = apply_stemming(text)
    text = rejoin_words(text)
    text = text + " " + " ".join(emoji)
    return text

def preprocess_DataFrame(df,Target_Column):
    target = Target_Column+"_Preprocessed"
    print("Start Preprocessed ENG")
    df[target] = df.progress_apply(lambda x: preprocess_Text(x[Target_Column]), axis=1)
    return df
