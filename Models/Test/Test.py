import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../../dataset/Train_model/Dataset_Train/Airline_Only_emoji', sep=',')
# Sentiment
encoder = LabelEncoder()
df['airline_sentiment'] = encoder.fit_transform(df['airline_sentiment'])

df['SEU_FINAL'] = np.where(df['SEU_FINAL'] < -0.05, 0,
                                 np.where(df['SEU_FINAL'] > 0.05, 2, 1))
# 'textblob-Compound' | 'Vader-Compound' | 'SEU_01'
y_true = df['airline_sentiment']
y_pred = df['SEU_FINAL']

print(classification_report(y_true, y_pred))

# definisci la mappatura dei valori
mapping = {0: 'negativo', 1: 'neutro', 2: 'positivo'}

# sostituisci i valori utilizzando la mappatura
df = df.replace({'airline_sentiment': mapping, 'SEU_FINAL': mapping})
y_true = df['airline_sentiment']
y_pred = df['SEU_FINAL']

labels = ["positivo", "negativo", "neutro"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

# crea un plot della matrice di confusione
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels,
       title="Matrice di Confusione",
       ylabel='Valore Reale',
       xlabel='Valore Predetto')

# ruota le etichette sull'asse x
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# aggiungi i valori alla matrice
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

# mostra il plot
fig.tight_layout()
plt.show()






