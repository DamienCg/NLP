import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from tqdm import tqdm
tqdm.pandas()
nltk.download('vader_lexicon')
from textblob import TextBlob

def TextBlob_fun(text):
    text = TextBlob(text)
    sentiment = text.sentiment
    return sentiment[0]

# Vader Sentiment!
def Vader_Compound_Fun(text):
    sentence = str(text)
    sid = SIA()
    sentiment = sid.polarity_scores(sentence)
    return sentiment["compound"]


text = 'i fli ❤️ ❤ ☺️ ☺ 👍'

print(TextBlob_fun(text))
print(Vader_Compound_Fun(text))
