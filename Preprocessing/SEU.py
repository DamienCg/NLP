import en_core_web_lg
import emoji as emj # pip install emoji==1.7
import pandas as pd
from transformers import pipeline
from tqdm import tqdm
tqdm.pandas()
from Standard_Preprocessing import *
EMOJIS = emj.UNICODE_EMOJI["en"]
Emoji_Compound = pd.read_csv('../dataset/Train_Model/Emoji_Sentiment_Train.csv', sep=',')
Emotion = pd.read_csv('../dataset/Train_Model/Italian-NRC-EmoLex.csv', sep=',')
negations = ["not", "n't","never", "no", "nothing", "neither", "nowhere", "none", "nt", "hardly", "scarcely",
                 "barely", "dont", "doesnt", "didnt", "shouldnt", "wouldnt", "couldnt", "cant", "wont", "isnt",
                 "wasnt", "aint", "havent", "hasnt", "hadnt", "doesnt", "didnt", "cant", "couldnt", "shouldnt",
                 "wouldnt", "wont", "aint"]


def find_negations(text):

    negations_found = []
    words = text.split()

    for i, word in enumerate(words):
        if word.lower() in negations:
            negations_found.append((i, word))

    return negations_found

def extract_emoji(text):
    list = []
    for emoji in EMOJIS:
        if emoji in text:
            list.append(emoji)
    return list

def Emoji_Compound_Fun(list_emoji):
    list = []
    if list_emoji is not None:
        for emoji in list_emoji:
            x = Emoji_Compound[(Emoji_Compound['Emoji'] == emoji)]
            if not x.empty:
                list.append(x['Compound'].values[0])
        if list:
            return float(sum(list) / len(list))
        else:
            return float(0.0)

def Extract_Emotion(text,lang='en'):
    list_Emotion = []
    # SEU_02
    positive_words = ['love', 'like', 'wonderful', 'awesome', 'great', 'excellent', 'good', 'fantastic',
                      'amazing', 'delightful', 'happy', 'joyful', 'smile', 'laughter',
                      'cheerful', 'brilliant', 'superb', 'terrific', 'perfect', 'glorious']
    # SEU_02 Tolgo anticipazione dai positivi
    Positive_Emotion = ['positive', 'joy']
    Negative_Emotion = ['negative', 'anger', 'disgust', 'fear', 'sadness']
    #SEU_02
    negations_list = find_negations(text)
    pos = 0
    neg = 0
    if lang == 'en':
        lang = 'English_Word'
    elif lang == 'it':
        lang = 'Italian_Word'
    # for each word in text
    for word in text.split():
        #SEU_02
        if word in positive_words:
            list_Emotion.append("positive")
            pos += 1

        x = Emotion[(Emotion[lang] == word.lower())]
        # print name of colum where value is 1
        if not x.empty:
            for i in range(0, len(x.columns)):
                if x.iloc[0, i] == 1:
                    list_Emotion.append(x.columns[i])
                    if x.columns[i] in Positive_Emotion:
                        pos += 1
                    if x.columns[i] in Negative_Emotion:
                        neg += 1
                        #SEU_02
                    if len(negations_list)>0:
                        list_Emotion.append("neg")
    return list_Emotion


def calculate_emotion_percentages(emotions):
    # Crea un dizionario per tenere traccia delle occorrenze di ogni emozione
    emotion_count = {}
    # Itera attraverso la lista delle emozioni
    for emotion in emotions:
        if emotion in emotion_count:
            # Incrementa il conteggio se l'emozione è già presente nel dizionario
            emotion_count[emotion] += 1
        else:
            # Imposta il conteggio su 1 se l'emozione non è presente nel dizionario
            emotion_count[emotion] = 1
    # Calcola la percentuale di ogni emozione
    total_emotions = len(emotions)
    result = []
    for emotion, count in emotion_count.items():
        percentage = (count / total_emotions) * 100
        result.append(f"{percentage:.2f}% {emotion}")
    return result


def Emotion_Compound_Fun(list):
    positive = 0
    negative = 0
    neutral = 1
    emotion_polarities = {
        "anger": -1,
        "positive": 2,
        "negative": -2,
        "disgust": -1,
        "fear": -1,
        "joy": 1,
        "sadness": -1,
        # SEU_02
        "Contrast": 0,
        "anticipation": 0,
        "surprise": 0,
        "trust": 0
    }
    for emotion in list:
        if emotion in emotion_polarities:
            if emotion_polarities[emotion] > 0:
                if emotion_polarities[emotion] >= 2:
                    positive += 2
                else:
                    positive += 1
            elif emotion_polarities[emotion] < 0:
                if emotion_polarities[emotion] <= -2:
                    negative += 2
                else:
                    negative += 1
        else:
            neutral += 1

    compound = (positive - negative) / (positive + negative + neutral)
    return float(compound)


def SEU(text, lang='en'):
    stops = set(stopwords.words("english"))
    emoji = extract_emoji(text)
    text = removetextuseless(text)
    text = tokenize(text)
    text = remove_stopwords(text, stops)
    text = rejoin_words(text)
    text = text + " " + " ".join(emoji)

    N = 0
    list_of_emoji = extract_emoji(text)
    if len(list_of_emoji) > 0:
        N += 1
    list_of_emotion = Extract_Emotion(text,lang)
    if len(list_of_emotion) > 0:
        N += 1
    if N == 0:
        N = 1
    compound_Emoji = Emoji_Compound_Fun(list_of_emoji)
    compound_Emotion = Emotion_Compound_Fun(list_of_emotion)


    compound = (compound_Emoji + compound_Emotion) / N

    Emotion_Percenages = calculate_emotion_percentages(list_of_emotion)
    return compound, Emotion_Percenages,list_of_emotion


a = SEU("text",lang='en')
print(a)


