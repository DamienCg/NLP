import spacy
import en_core_web_lg
import emoji as emj # pip install emoji==1.7
# Carica il modello di lingua inglese
nlp = spacy.load("en_core_web_sm")
from spacy.tokens import Token
from spacy.language import Language
from nltk import Tree
import pandas as pd
import re
EMOJIS = emj.UNICODE_EMOJI["en"]
import nltk
from nltk.corpus import sentiwordnet as swn
nltk.download('sentiwordnet')
import networkx as nx
import json
Emoji_Compound = pd.read_csv('../dataset/Train_Model/Emoji_Sentiment_Train.csv', sep=',')
Emotion = pd.read_csv('../dataset/Train_Model/Italian-NRC-EmoLex.csv', sep=',')
pos_map = {
    'ADJ': 'a',
    'ADP': 'r',
    'ADV': 'r',
    'AUX': 'v',
    'CCONJ': 'r',
    'DET': 'r',
    'INTJ': 'r',
    'NOUN': 'n',
    'NUM': 'r',
    'PART': 'r',
    'PRON': 'r',
    'PROPN': 'n',
    'PUNCT': 'r',
    'SCONJ': 'r',
    'SPACE': 'r',
    'SYM': 'r',
    'VERB': 'v',
    'X': 'r'
}
negations = ["not", "n't","never", "no", "nothing", "neither", "nowhere", "none", "nt", "hardly", "scarcely",
                 "barely", "dont", "doesnt", "didnt", "shouldnt", "wouldnt", "couldnt", "cant", "wont", "isnt",
                 "wasnt", "aint", "havent", "hasnt", "hadnt", "doesnt", "didnt", "cant", "couldnt", "shouldnt",
                 "wouldnt", "wont", "aint"]


def has_negation(text):
    global negations
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


import nltk

def Emotion_Compound_Fun(list):
    positive = 0
    negative = 0
    neutral = 1
    emotion_polarities = {
        "anger": -0.5,
        "positive": 1,
        "negative": -1,
        "disgust": -0.5,
        "fear": -0.5,
        "joy": 0.5,
        "sadness": -0.5,
        "anticipation": 0.0,
        "surprise": 0.0,
        "trust": 0.0
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

def removetextuseless(my_string):
    x = re.sub(r'http\S+', '', my_string)
    x = re.sub(r'www\S+', '', x)
    x = re.sub("@[A-Za-z0-9_]+", "", x)
    x = re.sub("#[A-Za-z0-9_]+", "", x)
    return x


def get_sentiment(word, pos):
    synsets = list(swn.senti_synsets(word, pos))
    if not synsets:
        return 0  # Sentiment not found for the given word and pos
    sentiment = sum(synset.pos_score() - synset.neg_score() for synset in synsets) / len(synsets)
    return sentiment

product = None
n_of_Children = 0
def multiply_children_float_values(node):
    global product
    global n_of_Children
    for child in node.children:
        if child._.float_value != 0:
            n_of_Children += 1
        if product is None:
            product = child._.float_value
        else:
            print("Sommo:"+str(product)+" + "+str(child._.float_value))
            if has_negation(child.text):
                print("has_neg")  # True
                product = -1
                return
            else:
                product = product + child._.float_value
            print("risultato: " +str(product))
        multiply_children_float_values(child) # chiamata ricorsiva sui figli
    if product is None:
        product = node._.float_value

def update_tree(token):
    global product
    global n_of_Children
    ROOT = token._.float_value
    Compound_List_of_Children = []
    if token.dep_ == 'ROOT':
        print("Sono sulla radice:")
        for child in token.children:
            product = None
            n_of_Children = 0
            print("eseguo somma ramo: " + child.text)
            multiply_children_float_values(child)
            flag_insert = False
            print("fine ramo:")
            print(n_of_Children)
            if product != None: #
                if n_of_Children > 0: # almeno un figlio
                    if product >= 1:
                        Compound_List_of_Children.append(1)
                        flag_insert = True
                    elif product <= -1:
                        Compound_List_of_Children.append(-1)
                        flag_insert = True
                    else:
                        product = product / n_of_Children
                        if ROOT != 0:
                            Compound_List_of_Children.append(product * ROOT)
                            flag_insert = True
                elif ROOT != 0:
                    Compound_List_of_Children.append(product * ROOT)
                    flag_insert = True

            print("Risultato: ", product)
            print("Lista compound figli: ", Compound_List_of_Children)

        if flag_insert == False:
            Compound_List_of_Children.append(ROOT)

    print("Stampo lista radice + figli:")
    print(Compound_List_of_Children)
    final_compound = 0
    valid = 0
    for child in Compound_List_of_Children:
        if child != 0:
            valid +=1
            final_compound += child

    if valid == 0:
        return ROOT
    return final_compound / valid




""" STAMPA """
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_




"""****************************************************************"""

def SEU(text, lang='en'):
    emoji = extract_emoji(text)
    text = removetextuseless(text)
    doc = nlp(text)

    @Language.component("set_float_value")
    def set_float_value(token):
        try:
            token._.float_value = float(0.0)
        except ValueError:
            pass

    Token.set_extension("float_value", default=None, force=True)
    nlp.add_pipe("set_float_value", name="set_float_value", first=True)

    for token in doc:
        emotion_scores = Extract_Emotion(token.text)
        compound_score = Emotion_Compound_Fun(emotion_scores)
        if(has_negation(token.text)):
            compound_score = -1
        if compound_score == 0 and compound_score != -1:
            wn_pos = pos_map[token.pos_]
            sentiment = get_sentiment(token.text, wn_pos)
            if sentiment >= 0.1 or sentiment <=-0.1:
                compound_score = sentiment
        token._.float_value = compound_score

    for token in doc:
        print(token.text, token._.float_value)

    def find_root(token):
        while token.head != token:
            token = token.head
        return token

    # utilizzo la radice del documento come punto di partenza
    root = find_root(doc[0].head)
    print("root: " + str(root))
    # eseguo la funzione ricorsiva
    Compound = update_tree(root)
    Emoji_Compound = Emoji_Compound_Fun(emoji)
    print("emoji compound")
    print(Emoji_Compound)
    if len(emoji) > 0:
        Compound = (Compound + Emoji_Compound) /2
    print("Risultato Compound:" + str(Compound))
    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


SEU("I don't know if i love tree", lang='en')