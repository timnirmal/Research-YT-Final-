from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import pandas as pd
import numpy as np
import torch
import re
import tqdm
from matplotlib._path import (affine_transform, count_bboxes_overlapping_bbox,
                              update_path_extents)

from gensim.models import Word2Vec
from nltk import word_tokenize, pos_tag, pos_tag_sents

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pandas.read_csv('../Riyana/Research/Text-Classification-Module-master/Dataset/testdata-domains.csv')
print(df.head(11))

df.groupby('Class')['Text'].nunique()


def clean_data(dataframe):
    # Drop duplicate rows
    dataframe.drop_duplicates(subset='Text', inplace=True)


clean_data(df)
df.to_csv('data/testdata-domains.csv')

df = pandas.read_csv('../Riyana/Research/Text-Classification-Module-master/Dataset/cleaned data-2.csv')
print(df.head(11))

trainDF = pandas.read_csv('../Riyana/Research/Text-Classification-Module-master/Dataset/cleaned data-2.csv')
print(trainDF['Filtered_sentence'].head(11))
trainDF['Filtered_sentence'] = trainDF['Filtered_sentence'].values.astype('U')

# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

train_x.nunique()

##### 2. Feature Engineering

####### 2.1 Count Vectors as features

# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['Filtered_sentence'])

# transform the training and validation data using count vectorizer object
xtrain_count = count_vect.transform(train_x)
xvalid_count = count_vect.transform(valid_x)

# Printing the identified Unique words along with their indices
print("Vocabulary: ", count_vect.vocabulary_)

# Encode the Document
vector = count_vect.transform(trainDF['Filtered_sentence'])

######## 2.2 TF-IDF Vectors as features

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['Filtered_sentence'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['Filtered_sentence'])
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram = tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                         max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['Filtered_sentence'])
xtrain_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(valid_x)

print("word level tf-idf_Vocabulary: ", tfidf_vect.vocabulary_)
print("ngram level tf-idf_Vocabulary: ", tfidf_vect_ngram.vocabulary_)
print("characters level tf-idf_Vocabulary: ", tfidf_vect_ngram_chars.vocabulary_)

print(xvalid_tfidf_ngram)


###########  3. Model Building

def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


###### 3.1 Multinominal Naive Bayes

# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", accuracy)

# Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy)

# Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy)

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, CharLevel Vectors: ", accuracy)

# Bernoulli Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.BernoulliNB(), xtrain_count, train_y, xvalid_count)
print("BernoulliNB, Count Vectors: ", accuracy)

# Bernoulli Naive Bayes on Word Level TF IDF Vectors
accuracy = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("BernoulliNB, WordLevel TF-IDF: ", accuracy)

# Bernoulli Naive Bayes on Ngram Level TF IDF Vectors
accuracy = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("BernoulliNB, N-Gram Vectors: ", accuracy)

# Bernoulli Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("BernoulliNB, CharLevel Vectors: ", accuracy)

############## 3.2 Logistic Regression-Linear Classifier

# Linear Classifier on Count Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("LR, CharLevel Vectors: ", accuracy)

############## 3.3 Support Vector Machine

# SVM on Count Vectors
accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)
print("SVM, Count Vectors: ", accuracy)

# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("SVM, Ngram Level TF IDF Vectors: ", accuracy)

# SVM on Word Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)
print("SVM, Word Level TF IDF Vectors: ", accuracy)

# SVM on Character Level TF IDF Vectors
accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("SVM, Character Level TF IDF Vectors: ", accuracy)

############ 3.4 Random Forest - Bagging Model

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print("RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print("RF, WordLevel TF-IDF: ", accuracy)

# RF on Ngram Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("RF, gram Level TF-IDF: ", accuracy)

# RF on Character Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("RF, Character Level TF IDF: ", accuracy)

############# KNN

from sklearn.neighbors import KNeighborsClassifier

# KNN on Count Vectors
accuracy = train_model(KNeighborsClassifier(n_neighbors=17, p=5, metric='euclidean'), xtrain_count, train_y,
                       xvalid_count)
print("KNN, Count Vectors: ", accuracy)

# KNN on Word Level TF IDF Vectors
accuracy = train_model(KNeighborsClassifier(n_neighbors=17, p=5, metric='euclidean'), xtrain_tfidf, train_y,
                       xvalid_tfidf)
print("KNN, WordLevel TF-IDF: ", accuracy)

# KNN on Ngram Level TF IDF Vectors
accuracy = train_model(KNeighborsClassifier(n_neighbors=200, p=5, metric='euclidean'), xtrain_tfidf_ngram, train_y,
                       xvalid_tfidf_ngram)
print("KNN, Ngram Level TF-IDF: ", accuracy)

# KNN on Character Level TF IDF Vectors
accuracy = train_model(KNeighborsClassifier(n_neighbors=12, p=5, metric='euclidean'), xtrain_tfidf_ngram_chars, train_y,
                       xvalid_tfidf_ngram_chars)
print("KNN, Character Level TF IDF: ", accuracy)


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


plt.rcParams["figure.figsize"] = (6, 5)

knn = KNeighborsClassifier(n_neighbors=7)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(xtrain_count, train_y)

no_neighbors = np.arange(1, 100)
train_accuracy = np.empty(len(no_neighbors))
test_accuracy = np.empty(len(no_neighbors))

for i, k in enumerate(no_neighbors):
    # We instantiate the classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit the classifier to the training data
    knn.fit(xtrain_count, train_y)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(xtrain_count, train_y)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(xtrain_count, train_y)

plt.title('k-NN: Varying Number of Neighbors')
plt.plot(no_neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(no_neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

########### 2.3 Text / NLP based features

# tokenization
from sinling import SinhalaTokenizer as tokenizer, SinhalaStemmer as stemmer, POSTagger, preprocess, word_joiner, \
    word_splitter
from collections import Counter
from itertools import chain

df = pandas.read_csv('../Riyana/Research/Text-Classification-Module-master/Dataset/testdata-domains.csv')
df['Text'] = df['Text'].apply(word_tokenize).tolist()

# POS tagging
tagger = POSTagger()
df['Text'] = tagger.predict(df['Text'].tolist())
df['Text'] = pd.Series(df['Text'].tolist())

# Use collections.Counter and itertools.chain to flatten the list of list-> get the POS vocabulary
tokens, tags = zip(*chain(*df['Text'].tolist()))
possible_tags = sorted(set(tags))
possible_tags_counter = Counter({p: 0 for p in possible_tags})

# Iterate through each tagged sentence and get the counts of POS
df['Text'].apply(lambda x: Counter(list(zip(*x))[1]))
df['pos_counts'] = df['Text'].apply(lambda x: Counter(list(zip(*x))[1]))
df['pos_counts']


# Add in the POS that don't appears in the sentence with 0 counts
def add_pos_with_zero_counts(counter, keys_to_add):
    for k in keys_to_add:
        counter[k] = counter.get(k, 0)
    return counter


# Flatten the values into the list
df['pos_counts'].apply(lambda x: add_pos_with_zero_counts(x, possible_tags))
df['pos_counts_with_zero'] = df['pos_counts'].apply(lambda x: add_pos_with_zero_counts(x, possible_tags))

df['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])

# Get the pos count vector
df['sent_vector'] = df['pos_counts_with_zero'].apply(lambda x: [count for tag, count in sorted(x.most_common())])

df['char_count'] = trainDF['Text'].apply(len)
df['word_count'] = trainDF['Text'].str.split().str.len()
df['word_density'] = df['char_count'] / (df['word_count'] + 1)
# trainDF['punctuation_count'] = trainDF['Filtered_sentence'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation)))
# trainDF['title_word_count'] = trainDF['Filtered_sentence'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
# trainDF['upper_case_word_count'] = trainDF['Filtered_sentence'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

pd.set_option('display.max_colwidth', None)
df

pd.set_option('display.max_colwidth', None)
df['sent_vector']

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import array

# pos_counts
data = []
for i in df["sent_vector"].values:
    temp = []
    for j in i:
        temp.append(j)
    data.append(temp)
data = np.array(data)

y = df['Class'].values

# RF
model = ensemble.RandomForestClassifier()
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.5, random_state=42)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('RF, pos_counts Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)))

# svm
model = svm.SVC()
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.5, random_state=42)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('svm, pos_counts Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)))

# LR
model = linear_model.LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.5, random_state=42)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('LR, pos_counts Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)))

# SGD
model = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.5, random_state=42)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('SGD, pos_counts Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)))

# NB
model = naive_bayes.MultinomialNB()
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.5, random_state=42)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('NB, pos_counts Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)))

############ Checked accuracy with balanced domain data


# RF
model = ensemble.RandomForestClassifier()
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.5, random_state=42)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('RF, pos_counts Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)))

# svm
model = svm.SVC()
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.5, random_state=42)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('svm, pos_counts Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)))

######### word count

# MultinomialNB
#	word count
X = df['word_count'].values
y = df['Class'].values

X = X.reshape(-1, 1)

model = naive_bayes.MultinomialNB()
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.8, random_state=100)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('NB, word_count Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)) * 100)

# RF
from sklearn.multiclass import OneVsRestClassifier

# Random forest
n_classes = 6

#	word_density
X = df['word_density'].values
y = df['Class'].values

X = X.reshape(-1, 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, y)

RF_clf = OneVsRestClassifier(ensemble.RandomForestClassifier())
RF_clf.fit(X_train, Y_train)
y_pred = RF_clf.predict(X_test)

print("Accuracy: ", RF_clf.score(X_test, Y_test))

from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import array

# pos_counts
data = []
for i in df["sent_vector"].values:
    temp = []
    for j in i:
        temp.append(j)
    data.append(temp)
data = np.array(data)

y = df['Class'].values

model = ensemble.RandomForestClassifier()
X_train, X_test, Y_train, Y_test = train_test_split(data, y, test_size=0.5, random_state=42)
# train_x, valid_x, train_y, valid_y = model.train_test_split(trainDF['Filtered_sentence'], trainDF['Class'])

model.fit(X_train, Y_train)

predictions = model.predict(X_test)
print('RF, pos_counts Accuracy: ', accuracy_score(Y_test.astype(str), predictions.astype(str)))
