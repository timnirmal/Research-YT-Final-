Host = "colab"  # @param ["colab", "AWS", "GCP"]

Account = "colab_datapirates"  # @param["colab_datapirates", "colab_lahiru_cse", "colab_lahiru_personal"]
EMBEDDING_SIZE = 300  # @param [50, 150, 200, 250, 300, 350, 400, 450, 500]
embedding_type = "fasttext"  # @param ["fasttext","word2vec"]
experiment_no = "1001"  # @param [] {allow-input: true}
model_type = "LSTM"  # @param ["RNN","GRU", "LSTM", "BiLSTM" ]

stack_modeles = ""  # @param ["","2","3"]
apply_CNN = False  # @param {type:"boolean"}

model_name = model_type + "_model"
if (stack_modeles == "2" or stack_modeles == "3"):
    model_name = "stacked_" + model_name + "_" + stack_modeles
if (apply_CNN):
    model_name = "CNN_" + model_name

print(model_name)

import collections
import pickle
import re
import random
import sys
import os
import time

import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastText
from gensim.models import word2vec

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix, precision_recall_fscore_support

import pandas as pd
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros
from numpy import cumsum

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dropout, Activation, Flatten, Embedding, Convolution1D, MaxPooling1D, AveragePooling1D, Input, Dense, Add, TimeDistributed, Bidirectional, SpatialDropout1D
from keras.layers import merging
from keras.layers import LSTM, GRU, SimpleRNN
from keras.regularizers import l2, l1_l2
from keras.constraints import maxnorm
from keras import callbacks
from keras.utils import generic_utils, plot_model
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.image as mpimg
import matplotlib.pyplot as plt



all_data = pd.read_csv("tagged_comments.csv")
print(all_data.shape)


# edit this later
def text_preprocessing(train_data, test_data):
    train_data_texts = train_data['comment']
    train_data_labels = train_data['label']
    test_data_texts = test_data['comment']
    test_data_labels = test_data['label']

    comment_texts = []
    comment_labels = []

    train_text = []
    test_text = []
    train_labels = []
    test_labels = []

    for comment in train_data_texts:
        lines = []
        try:
            words = comment.split()
            lines += words
        except:
            continue
        train_text.append(lines)
    comment_texts.append(train_text)

    for comment in test_data_texts:
        lines = []
        try:
            words = comment.split()
            lines += words
        except:
            continue
        test_text.append(lines)
    comment_texts.append(test_text)

    return comment_texts, comment_labels


# comment_texts, comment_labels = text_preprocessing(all_data, all_data)
#
# print(comment_texts)
# print(comment_labels)


# edit this later
def text_preprocessing_1(data):
    comments = data['comment']
    labels = data['label']

    comments_splitted = []
    labels_encoded = []

    for label in labels:
        if label == "POSITIVE":
            labels_encoded.append(1)
        else:
            labels_encoded.append(0)

    for comment in comments:
        lines = []
        try:
            words = comment.split()
            lines += words
        except:
            continue
        comments_splitted.append(lines)
    return comments_splitted, labels_encoded


def text_preprocessing_2(data):
    comments = data['comment']
    labels = data['label']

    comments_splitted = []

    for comment in comments:
        lines = []
        try:
            words = comment.split()
            lines += words
        except:
            continue
        comments_splitted.append(lines)

    return comments_splitted, labels


comment_texts, comment_labels = text_preprocessing_2(all_data)


# prepare tokenizer

t = Tokenizer()
t.fit_on_texts(comment_texts)
vocab_size = len(t.word_index) + 1
print(vocab_size)

encoded_docs = t.texts_to_sequences(comment_texts)
max_length = len(max(encoded_docs, key=len))
padded_docs = pad_sequences(encoded_docs, maxlen=max_length)

comment_labels = np.array(comment_labels)
padded_docs = np.array(padded_docs)
comment_labels = pd.get_dummies(comment_labels).values
print('Shape of label tensor:', comment_labels.shape)

X_train, X_test, y_train, y_test = train_test_split(padded_docs, comment_labels, test_size=0.1, random_state=0)
(unique, counts) = np.unique(y_test, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print(frequencies)

print(X_train)
print(y_train)
print(X_test)
print(y_test)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


def generate_embedding_metrix():
    if (embedding_type == 'fasttext'):
        word_embedding_model = FastText.load(word_embedding_path)
    else:
        word_embedding_model = word2vec.Word2Vec.load(word_embedding_path)

    word_vectors = word_embedding_model.wv
    word_vectors.save(word_embedding_keydvectors_path)
    word_vectors = KeyedVectors.load(word_embedding_keydvectors_path, mmap='r')

    embeddings_index = dict()
    for word, vocab_obj in word_vectors.vocab.items():
        embeddings_index[word] = word_vectors[word]

    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, embedding_size))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    pickle.dump(embedding_matrix, open(embedding_matrix_path, 'wb'))
    return embedding_matrix



# context = 5
# word_embedding_path = folder_path + "word_embedding/"+embedding_type+"/source2_data_from_gosspiLanka_and_lankadeepa/"+str(EMBEDDING_SIZE)+"/"+embedding_type+"_"+str(EMBEDDING_SIZE)+"_"+str(context)
# word_embedding_keydvectors_path = folder_path + "word_embedding/"+embedding_type+"/source2_data_from_gosspiLanka_and_lankadeepa/"+str(EMBEDDING_SIZE)+"/keyed_vectors/keyed.kv"
# embedding_matrix_path = folder_path + 'Sentiment Analysis/CNN RNN/embedding_matrix/'+embedding_type+'_lankadeepa_gossiplanka_'+str(EMBEDDING_SIZE)+'_'+str(context)
#
# experiment_name = folder_path + "Sentiment Analysis/CNN RNN/experiments/" +str(experiment_no) + "_"+ model_name +"_"+embedding_type+"_"+str(EMBEDDING_SIZE)+"_"+str(context)
# model_save_path = folder_path + "Sentiment Analysis/CNN RNN/saved_models/"+str(experiment_no)+"_weights_best_"+model_name+"_"+embedding_type+"_"+str(experiment_no)+".hdf5"
#


def load_word_embedding_atrix():
    f = open(embedding_matrix_path, 'rb')
    embedding_matrix = np.array(pickle.load(f))
    return embedding_matrix


def RNN_model(RNN_layer, maxlen, hidden_dims, l2_reg, drop_out_value_1, drop_out_value_2):
    main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
    embedding = Embedding(MAX_FEATURES, EMBEDDING_SIZE,
                          weights=[EMBEDDING_MATRIX], input_length=maxlen,
                          name='embedding', trainable=False)(main_input)

    embedding = Dropout(drop_out_value_1)(embedding)

    x = RNN(hidden_dims)(embedding)

    x = Dense(hidden_dims, activation='relu', init='he_normal',
              W_consඔබගේ දැනුමට කරන්ඩ ඕන ඒක මට අදාළ කාරණාවක් නෑtraint=maxnorm(3), b_constraint=maxnorm(3),
              name='mlp')(x)

    x = Dropout(drop_out_value_2, name='drop')(x)

    output = Dense(4, init='he_normal',
                   activation='softmax', name='output')(x)

    model = Model(input=main_input, output=output, name="RNN_model")

    model.compile(loss={'output': 'categorical_crossentropy'},
                  optimizer=Adadelta(lr=0.95, epsilon=1e-06),
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           f1])

    print(model.summary())
    return model


def stacked_RNN_model_2(RNN_layer, maxlen, hidden_dims, l2_reg, drop_out_value_1, drop_out_value_2):
    main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
    embedding = Embedding(MAX_FEATURES, EMBEDDING_SIZE,
                          weights=[EMBEDDING_MATRIX], input_length=maxlen,
                          name='embedding', trainable=False)(main_input)

    embedding = Dropout(drop_out_value_1)(embedding)

    x = RNN_layer(hidden_dims, return_sequences=True)(embedding)
    x = RNN_layer(hidden_dims)(x)

    x = Dense(hidden_dims, activation='relu', init='he_normal',
              W_constraint=maxnorm(3), b_constraint=maxnorm(3),
              name='mlp')(x)

    x = Dropout(drop_out_value_2, name='drop')(x)

    output = Dense(4, init='he_normal',
                   activation='softmax', name='output')(x)

    model = Model(input=main_input, output=output, name="stacked_RNN_model_2")

    model.compile(loss={'output': 'categorical_crossentropy'},
                  optimizer=Adadelta(lr=0.95, epsilon=1e-06),
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           f1])

    print(model.summary())
    return model


def stacked_RNN_model_3(RNN_layer, maxlen, hidden_dims, l2_reg, drop_out_value_1, drop_out_value_2):
    main_input = Input(shape=(maxlen,), dtype='int32', name='main_input')
    embedding = Embedding(MAX_FEATURES, EMBEDDING_SIZE,
                          weights=[EMBEDDING_MATRIX], input_length=maxlen,
                          name='embedding', trainable=False)(main_input)

    embedding = Dropout(drop_out_value_1)(embedding)

    x = RNN_layer(hidden_dims, return_sequences=True)(embedding)
    x = RNN_layer(hidden_dims, return_sequences=True)(x)
    x = RNN_layer(hidden_dims)(x)

    x = Dense(hidden_dims, activation='relu', init='he_normal',
              W_constraint=maxnorm(3), b_constraint=maxnorm(3),
              name='mlp')(x)

    x = Dropout(drop_out_value_2, name='drop')(x)

    output = Dense(4, init='he_normal',
                   activation='softmax', name='output')(x)

    model = Model(input=main_input, output=output, name="stacked_RNN_model_3")

    model.compile(loss={'output': 'categorical_crossentropy'},
                  optimizer=Adadelta(lr=0.95, epsilon=1e-06),
                  metrics=["accuracy",
                           tf.keras.metrics.Precision(),
                           tf.keras.metrics.Recall(),
                           f1])

    print(model.summary())
    return model



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


