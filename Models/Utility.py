import re
import nltk
from Models.Load_dataset import *
import emoji as emj # pip install emoji==1.7
EMOJIS = emj.UNICODE_EMOJI["en"]
from Models.Hyper_param_SEU_02 import emotion_polarities
from nltk import Tree
from Models.list import *

def removetextuseless(my_string):
    x = re.sub(r'http\S+', '', my_string)
    x = re.sub(r'www\S+', '', x)
    x = re.sub("@[A-Za-z0-9_]+", "", x)
    x = re.sub("#[A-Za-z0-9_]+", "", x)
    return x

def has_negation(text):
    # Tokenizza il testo in parole
    words = nltk.word_tokenize(text)
    if text in negations:
        return True
    # Costruisce una lista di espressioni regolari che corrispondono alle parole che indicano negazione
    negation_words = ['not', 'n\'t', 'never', 'no', 'nothing', 'nowhere', 'noone', 'none']
    negation_patterns = [r'(?:^|\s)%s(?:$|\s)' % w for w in negation_words]

    # Crea una espressione regolare che cerca negazioni all'interno della frase
    negation_regex = re.compile('|'.join(negation_patterns), re.IGNORECASE)

    # Cerca se la frase contiene una negazione
    return bool(negation_regex.search(text))


def extract_emoji(text):
    list = []
    for emoji in EMOJIS:
        if emoji in text:
            list.append(emoji)
    return list

def has_emoji(text):
    for emoji in EMOJIS:
        if emoji in text:
            return True
    return False

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

def Emotion_Compound_Fun(list):
    positive = 0
    negative = 0
    neutral = 1
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



def Extract_Emotion(text,lang='en'):
    list_Emotion = []
    if lang == 'en':
        lang = 'English_Word'
    elif lang == 'it':
        lang = 'Italian_Word'

    # for each word in text
    for word in text.split():
        x = Emotion[(Emotion[lang] == word.lower())]
        # print name of colum where value is 1
        if not x.empty:
            for i in range(0, len(x.columns)):
                if x.iloc[0, i] == 1:
                    list_Emotion.append(x.columns[i])

    return list_Emotion


""" STAMPA """
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_