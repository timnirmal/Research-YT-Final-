import pickle

import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec, word2vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import pad_sequences

with open('models/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

word2vec_model = word2vec.Word2Vec.load("../../hate/embedding/word2vec_300.w2v")

# Preprocessing
embedding_dim = 300
max_length = 20
trunc_type = 'post'
padding_type = 'post'
training_size = 6000
test_portion = .1


# Define the LSTM model
model = Sequential()
model.add(Embedding(len(word2vec_model.wv.key_to_index), embedding_dim, input_length=max_length, trainable=False))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# load model weights
model.load_weights('models/lstm_model_weights.h5')



def process_text(text):
    # Tokenize the text into words
    words = text.split()

    # Convert words to word indices
    word_indices = []
    for word in words:
        if word in word2vec_model.wv.key_to_index:
            word_indices.append(word2vec_model.wv.key_to_index[word])

    # Pad the sequence to ensure consistent length
    padded_sequences = pad_sequences([word_indices], maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return padded_sequences

def predict_sentiment(text):
    # # load X_test
    # with open('X_test.pkl', 'rb') as file:
    #     X_test = pickle.load(file)
    #
    # # Make predictions
    # predictions = model.predict(X_test)
    # classes = np.argmax(predictions, axis=1)
    #
    # print("Predictions: ", predictions)
    # print("Classes: ", classes)
    #
    # Process the text data
    processed_text = process_text(text)

    # # save processed text
    # with open('models/processed_text.pkl', 'wb') as file:
    #     pickle.dump(processed_text, file)
    #
    # # load processed text
    # with open('models/processed_text.pkl', 'rb') as file:
    #     processed_text = pickle.load(file)

    # print("processed_text", processed_text)
    # print("processed_text.shape", processed_text[0])
    # print("processed_text.shape", processed_text[0][0])

    # Make the prediction
    predictions = model.predict(processed_text)

    # print("predicted_class", predictions)

    if predictions[0] < 0.5:
        predicted_label = 'Not Hate Speech'
        predicted_label_num = 0
        print("Not Hate Speech")
        print('0')
    else:
        predicted_label = 'Hate Speech'
        predicted_label_num = 1
        print("Hate Speech")
        print('1')

    return predicted_label_num

def predict_sentiment_df(df):
    df['hate'] = df['text'].apply(lambda x: predict_sentiment(x))

    return df


def evaluate_model():
    # load X_test
    with open('models/X_test.pkl', 'rb') as file:
        X_test = pickle.load(file)

    # load y_test
    with open('models/y_test.pkl', 'rb') as file:
        y_test = pickle.load(file)

    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy: %.2f%%' % (scores[1] * 100))

if __name__ == '__main__':
    text = "25 කෙල්ල අයියලගෙ කාල"
    sentiment = predict_sentiment(text)
    print("Predicted sentiment:", sentiment)

    text = "25 එකෙක් ඇරියා වගේ"
    sentiment = predict_sentiment(text)
    print("Predicted sentiment:", sentiment)

    text = "දරුව මගෙ කියලා හිතුන මම"
    sentiment = predict_sentiment(text)
    print("Predicted sentiment:", sentiment)

    evaluate_model()



