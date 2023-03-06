import spacy
from spacy.tokens import Token
from spacy.language import Language
import nltk
nltk.download('sentiwordnet')
from Models.Utility_Tree_Core_SEU_02 import *
from tqdm.auto import tqdm
tqdm.pandas()
from Hyper_param_SEU_02 import *
from tqdm import tqdm
tqdm.pandas()


def SEU(text, lang='en',print_tree=False,verbose=False):
    emoji = extract_emoji(text)
    text = removetextuseless(text)
    nlp_en = spacy.load("en_core_web_sm")
    doc = nlp_en(text)

    list_of_emotion = []

    @Language.component("set_float_value")
    def set_float_value(token):
        try:
            token._.float_value = float(0.0)
        except ValueError:
            pass

    Token.set_extension("float_value", default=None, force=True)
    nlp_en.add_pipe("set_float_value", name="set_float_value", first=True)


    for token in doc:
        emotion_scores = Extract_Emotion(token.text)
        for i in emotion_scores:
            list_of_emotion.append(i)

        compound_score = Emotion_Compound_Fun(emotion_scores)
        if(has_negation(token.text)):
            compound_score = -1
        if compound_score == 0 and compound_score != -1:
            wn_pos = pos_map[token.pos_]
            sentiment = get_sentiment(token.text, wn_pos)
            if sentiment >= 0.1 or sentiment <=-0.1:
                compound_score = sentiment
        token._.float_value = compound_score
        if verbose == True:
            print(str(token.text)+ " sentiment-> "+str(token._.float_value))

    if verbose == True:
        print("")
        print("**********************")
        print("")

    def find_root(token):
        while token.head != token:
            token = token.head
        return token

    root = find_root(doc[0].head)
    Compound = update_tree(root,verbose)
    Emoji_Compound = Emoji_Compound_Fun(emoji)
    print(Emoji_Compound)
    if len(emoji) > 0:
        Compound = ((Compound*Peso_text) + (Emoji_Compound*Peso_emoticon)) /1
    if print_tree == True:
        [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
    return Compound,list_of_emotion


print(SEU("They told me no, but I love nature  😍",print_tree=True,verbose=True))


"""df = pd.read_csv('../dataset/Train_model/Dataset_Train/Airline_Only_emoji', sep=',')
# remove to df column text nan
df = df.dropna(subset=['text'])
df['SEU_FINAL'] = df['text'].progress_apply(lambda x: SEU(x)[0])
# save df to csv
df.to_csv('../dataset/Train_model/Dataset_Train/Airline_Only_emoji', sep=',', index=False)
"""
