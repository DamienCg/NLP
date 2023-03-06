from nltk.corpus import sentiwordnet as swn
from Models.Utility import *

def get_sentiment(word, pos):
    synsets = list(swn.senti_synsets(word, pos))
    if not synsets:
        return 0  # Sentiment not found for the given word and pos
    sentiment = sum(synset.pos_score() - synset.neg_score() for synset in synsets) / len(synsets)
    return sentiment

def Somma_albero(node):
    somma_positivi = 0
    peso_positivi = 0
    somma_negativi = 0
    peso_negativi = 0
    num_positivi = 0
    num_negativi = 0
    flag_is_entry = False
    ROOT = node._.float_value

    for node in node.children:
        flag_is_entry = True
        if node._.float_value > 0:
            somma_positivi += node._.float_value * node._.float_value
            peso_positivi += node._.float_value
            num_positivi += 1
        elif node._.float_value < 0:
            if node._.float_value == -1:
                return -1
            somma_negativi += node._.float_value * (-node._.float_value)
            peso_negativi += (-node._.float_value)
            num_negativi += 1

    if flag_is_entry == False:
        return ROOT

    if ROOT > 0:
        somma_positivi += ROOT * ROOT
        peso_positivi += ROOT
        num_positivi += 1
    elif ROOT < 0:
        if ROOT == -1:
            return -1
        somma_negativi += ROOT * (-ROOT)
        peso_negativi += (-ROOT)
        num_negativi += 1

    if num_positivi == 0:
        media_positivi = 0
    else:
        media_positivi = somma_positivi / peso_positivi if peso_positivi > 0 else 0

    if num_negativi == 0:
        media_negativi = 0
    else:
        media_negativi = somma_negativi / peso_negativi if peso_negativi > 0 else 0

    if num_positivi == 0 and num_negativi == 0:
        media = 0
    elif num_positivi == 0:
        media = media_negativi
    elif num_negativi == 0:
        media = media_positivi
    else:
        media = (media_positivi + media_negativi) / 2

    return media

def right_value(Compound_List_of_Children):
    ret = 0
    for i in Compound_List_of_Children:
        if i >= 0.1 or i <= -0.1:
            ret += 1
    return ret

def update_tree(token,verbose=False):
    ROOT = token._.float_value
    Compound_List_of_Children = []
    if token.dep_ == 'ROOT':
        for child in token.children:
            if verbose == True:
                print("Sommo ramo: "+str(child.text))
            product = Somma_albero(child)
            if verbose == True:
                print("Risultato somma: " + str(product))
            if product != 0:
                Compound_List_of_Children.append(product)

    n = right_value(Compound_List_of_Children)
    if verbose == True:
        print("Lista Prima delle op finali: ")
        print(Compound_List_of_Children)
    if -1 in Compound_List_of_Children:
        for index, value in enumerate(Compound_List_of_Children):
            if value == -1:
                Compound_List_of_Children[index] = ROOT*(-1)
        if verbose == True:
            print("Lista con aggiunta di Root * (-1) ")
            print(Compound_List_of_Children)
        return sum(Compound_List_of_Children) / n
    elif len(Compound_List_of_Children) > 0 and sum(Compound_List_of_Children) != 0:
        Compound_List_of_Children.append(ROOT)
        if verbose == True:
            print("Lista con la radice aggiunta operazione sum(list)/len(lista): ")
            print(Compound_List_of_Children)
        if n > 0:
            return sum(Compound_List_of_Children)/(n)
        else:
            return ROOT
    else:
        if verbose == True:
            print("Ritorno solo il valore della radice")
        return ROOT

