from Standard_Preprocessing import *
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from textblob import TextBlob
from tqdm import tqdm
tqdm.pandas()
import pandas as pd
nltk.download('vader_lexicon')


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

df = pd.read_csv('../dataset/Train_model/Dataset_Train/T4SA_Reducted_Sentiment_SEU_02.csv', sep=',')
# remove to df column text nan
#df = df.dropna(subset=['text'])
#df = preprocess_DataFrame(df,"text")
df['Vader-Compound'] = df['text'].progress_apply(lambda x: Vader_Compound_Fun(x))
df['textblob-Compound'] = df['text'].progress_apply(lambda x: TextBlob_fun(x))
# save df to csv

#df = df[['text','text_Preprocessed','airline_sentiment','Vader-Compound','textblob-Compound']]
df.to_csv('../dataset/Train_model/Dataset_Train/T4SA_Reducted_Sentiment_SEU_02.csv', sep=',', index=False)