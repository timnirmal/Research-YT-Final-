import pickle

import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec, word2vec
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import pad_sequences

data = pd.read_csv('sinhala-hate-speech-dataset.csv')  # Replace with your actual file path
texts = data['comment'].tolist()
labels = data['label'].tolist()

# Tokenize the text into sentences and words
sentences = [text.split() for text in texts]

# Preprocessing
embedding_dim = 300
max_length = 20
trunc_type = 'post'
padding_type = 'post'

# load word2vec model
word2vec_model = word2vec.Word2Vec.load("../../hate/embedding/word2vec_300.w2v")

# Convert sentences to sequences of word indices
word_indices = []
for sentence in sentences:
    indices = []
    for word in sentence:
        if word in word2vec_model.wv.key_to_index:
            indices.append(word2vec_model.wv.key_to_index[word])
        # else:
        #     indices.append(word2vec_model.wv.key_to_index[''])
    word_indices.append(indices)


# Pad sequences to ensure consistent length
padded_sequences = pad_sequences(word_indices, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(word2vec_model.wv.key_to_index), embedding_dim, input_length=max_length, trainable=False))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

# # rewite model as x = model()
# x = Embedding(len(word2vec_model.wv.key_to_index), embedding_dim, input_length=max_length, trainable=False)(input)
# model_x = Model(inputs=X_train, outputs=x)
#
# y = Dropout(0.2)(x)
# y = LSTM(100)(y)
# y = Dense(1, activation='sigmoid')(y)
# model_y = Model(inputs=model_x, outputs=y)

# Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# save X_train, X_test, y_train, y_test
pickle.dump(X_train, open('models/X_train.pkl', 'wb'))
pickle.dump(X_test, open('models/X_test.pkl', 'wb'))
pickle.dump(y_train, open('models/y_train.pkl', 'wb'))
pickle.dump(y_test, open('models/y_test.pkl', 'wb'))

# load
# X_train = pickle.load(open('models/X_train.pkl', 'rb'))
# X_test = pickle.load(open('models/X_test.pkl', 'rb'))
# y_train = pickle.load(open('models/y_train.pkl', 'rb'))
# y_test = pickle.load(open('models/y_test.pkl', 'rb'))

my_callbacks = [
    EarlyStopping(patience=20),
    # ModelCheckpoint(filepath='model_chk/model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True),
]

# Train the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1, callbacks=[my_callbacks])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)


model.save_weights('models/lstm_model_weights.h5')
model.save('models/lstm_model.h5')
#
# import joblib
#
# # save the model
# joblib.dump(model, 'models/lstm_model_joblib.pkl')

# save the label encoder
pickle.dump(label_encoder, open('models/label_encoder.pkl', 'wb'))

# load the model
# model = load_model('lstm_model/lstm.h5')

# load the label encoder
# label_encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))



# Evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: %.2f%%' % (scores[1] * 100))

# Make predictions
predictions = model.predict(X_test)
classes = np.argmax(predictions, axis=1)

print("Predictions: ", predictions)
print("Classes: ", classes)

# # Invert the predictions
# inverted_predictions = label_encoder.inverse_transform(predictions.reshape(1, -1)[0])

# # Save the predictions to a CSV file
# output = pd.DataFrame({'comment': texts, 'label': labels, 'prediction': inverted_predictions})
# output = pd.DataFrame({'comment': texts, 'label': labels, 'prediction': classes})

# output.to_csv('predictions.csv', index=False)