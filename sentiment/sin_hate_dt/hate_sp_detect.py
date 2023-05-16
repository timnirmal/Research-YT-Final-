# Imports
import tensorflow as tf
import csv
import random
import numpy as np
import pandas as pd
import io
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

# Reading the dataset
dataset = []
with open("sinhala-hate-speech-dataset.csv", encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        label = row[2]
        post = row[1]
        item = []
        item.append(post)
        if label == '0':
            item.append(0)
        elif label == '1':
            item.append(1)
        else:
            continue
        dataset.append(item)

posts = []
labels = []
random.shuffle(dataset)
for x in range(len(dataset)):
    posts.append(dataset[x][0])
    labels.append(dataset[x][1])

# Splitting the dataset
training_posts = posts[0:6000]
training_labels = labels[0:6000]
evaluation_posts = posts[6000:6345]
evaluation_labels = labels[6000:6345]

# Preprocessing
embedding_dim = 300
max_length = 20
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 6000
test_portion = .1

tokenizer = Tokenizer()
tokenizer.fit_on_texts(training_posts)
word_index = tokenizer.word_index
vocab_size = len(word_index)
sequences = tokenizer.texts_to_sequences(training_posts)
padded = pad_sequences(sequences, maxlen=max_length,
                       padding=padding_type, truncating=trunc_type)

# Splitting the dataset into training and validation
split = int(test_portion * training_size)
test_sequences = padded[0:split]
training_sequences = padded[split:training_size]
test_labels = training_labels[0:split]
training_labels = training_labels[split:training_size]

print("Vocab size: ", vocab_size)

# Loading pretrained FastText word embeddings in Sinhala
from gensim.models import FastText, word2vec

w2v_model = word2vec.Word2Vec.load("../../hate/embedding/word2vec_300.w2v")

print("Embedding dim: ", embedding_dim)

# Mapping FastText word vectors with word in the dataset
embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim));

# count items in word_index.items()
item_in_word_index = len(word_index.items())
print("Total items in word_index: ", item_in_word_index)

for word, i in word_index.items():
    print("Word: ", word)
    print("Index: ", i)
    if word in w2v_model.wv.key_to_index:
        print("Word vector: ", w2v_model.wv[word])
        embedding_vector = w2v_model.wv[word];
    else:
        print("Word vector: ", None)
        embeddings_matrix[i] = embedding_vector;

    # if embedding_vector is not None:

print("Embeddings matrix: ", embeddings_matrix)
print("Label: ", training_labels)

print("Embeddings matrix shape: ", embeddings_matrix.shape)

from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(embeddings_matrix, training_labels)

#
# # Training the model
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size + 1, embedding_dim,
#                               input_length=max_length, weights=[embeddings_matrix],
#                               trainable=False),
#     tf.keras.layers.LSTM(128),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(vocab_size + 1, embedding_dim,
#                               input_length=max_length, weights=[embeddings_matrix],
#                               trainable=False))
# model.add(tf.keras.layers.LSTM(128))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



print("Model created")

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model compiled")
model.summary()
num_epochs = 10

training_padded = np.array(training_sequences)
training_labels = np.array(training_labels)
testing_padded = np.array(test_sequences)
testing_labels = np.array(test_labels)

# save the tokenizer
import pickle

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save


print("Training the model")

history = model.fit(training_padded, training_labels,
                    epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)
print("Training Complete")

# Plotting accuracies and errors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))  # Get number of epochs

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])
plt.figure()

print("Evaluating the model")
# Model Evaluation
eval_sequences = tokenizer.texts_to_sequences(evaluation_posts)
eval_padded = pad_sequences(eval_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
pred = model.predict(eval_padded)
predicted_labels = []
for x in pred:
    predicted_labels.append(int(x.round().item()))

predicted_labels = tf.convert_to_tensor(predicted_labels)
from sklearn.metrics import classification_report

print(classification_report(evaluation_labels, predicted_labels))


# Testing model with sample text
def hate_speech(post):
    post_sequence = tokenizer.texts_to_sequences(post)
    padded_post_sequence = pad_sequences(post_sequence,
                                         maxlen=max_length, padding=padding_type,
                                         truncating=trunc_type)
    post_prediction = model.predict(padded_post_sequence)
    print(post_prediction)
    print(post_prediction.round())
    print(post_prediction.round().item())
    label = post_prediction.round().item()

    if label == 0:
        print("%s : Post is NOT Hate speech" % post)
    elif label == 1:
        print("%s : Post is Hate speech" % post)


hate_speech(['අද හොඳ දවසක්'])
hate_speech(['ගොන් බුරුවා කන පලාගන්න එපා'])
hate_speech(['පලයන් තම්බියා'])
hate_speech(['මම එහෙම දෙයක් කිව්වෙ නෑ මචන්'])

# Saving the model
model.save("hate_speech_model.h5")
print("Model saved")
