import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# NLP Libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Deep Learning Libraries
import tensorflow as tf
from keras.layers import Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.preprocessing.text import one_hot
from keras.layers import Bidirectional

# Vocabulary Size
voc_size = 5000
df = pd.read_csv(r"D:\Projects\Research-YT\Research-YT-Final\Process_Audio\New_Hate_Sentiment\Sinhala_Singlish_Hate_Speech_Processed.csv")

# print columns
print(df.columns)

# keep only Text_cleaned and label
df = df[['Text_cleaned', 'IsHateSpeech']]

# rename Text_cleaned and IsHateSpeech
df = df.rename(columns={'Text_cleaned': 'title', 'IsHateSpeech': 'label'})


# rename label to YES -> 1 and NO -> 0
df['label'] = df['label'].replace(['YES', 'NO'], [1, 0])

print(df.head())

df = df.dropna()
x = df.drop('label', axis=1)
y = df['label']

var = x.copy()
var.reset_index(inplace=True)

# DOWNLOADING THE STOPWORDS
nltk.download('stopwords')

# TEXT PREPROCESSING
# STEMMING, REMOVAL OF STOP WORDS , CONVERTING INTO LOWER CASE
stemmer = PorterStemmer()
corpus = []
for i in range(0, len(var)):
    review = re.sub('[^a-zA-Z]', ' ', var['title'][i])
    review = review.lower()
    review = review.split()

    review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

corpus[:5]

encoding = [one_hot(words, voc_size) for words in corpus]
emb_docs = pad_sequences(encoding, maxlen=20, padding='pre')
print(emb_docs)

model = Sequential()
model.add(Embedding(voc_size, 40, input_length=20))  # making embedding layer
model.add(Bidirectional(LSTM(100)))  # one LSTM Layer with 100 neurons
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
x_final = np.array(emb_docs)
y_final = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.2, random_state=0)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

# prediction
y_pred = model.predict(x_test)
print(y_pred)
classes_pred = np.argmax(y_pred, axis=1)
print(classes_pred)

print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, y_pred))
