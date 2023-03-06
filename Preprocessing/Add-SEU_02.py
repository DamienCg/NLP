import pandas as pd
from Models.SEU_02 import *
from tqdm.auto import tqdm
tqdm.pandas()

df = pd.read_csv('../dataset/Train_model/Dataset_Train/Airline_Sentiment_Preprocessed.csv', sep=',')
# remove to df column text nan
df = df.dropna(subset=['text'])
df['SEU_2'] = df['text'].progress_apply(lambda x: SEU(x)[0])
# save df to csv
df.to_csv('../dataset/Train_model/Dataset_Train/Airline_Sentiment_Prepocessed.csv', sep=',', index=False)