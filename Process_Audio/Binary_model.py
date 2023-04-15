from joblib import dump, load

import pandas as pd
import numpy as np
import nltk
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.externals import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import load_model
import os.path
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import class_weight
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

# show all columns
pd.set_option('display.max_columns', None)


################################################## Multinomial Naive Bayes model

def nb_train(X_train, X_test, y_train, y_test):
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB())])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return nb, accuracy, report


# KNN
def knn_train(X_train, X_test, y_train, y_test):
    knn = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', KNeighborsClassifier(n_neighbors=17, p=6, metric='euclidean'))])
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return knn, accuracy, report


# RF
from sklearn import decomposition, ensemble


def RF_train(X_train, X_test, y_train, y_test):
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', ensemble.RandomForestClassifier())])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return nb, accuracy, report


# LR
from sklearn import linear_model


def LR_train(X_train, X_test, y_train, y_test):
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', linear_model.LogisticRegression())])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return nb, accuracy, report


# Svm
def svc_train(X_train, X_test, y_train, y_test):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', svm.SVC()),
                    ])
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return sgd, accuracy, report


# SGD
def svc_train(X_train, X_test, y_train, y_test):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return sgd, accuracy, report


from sklearn import decomposition, ensemble

tags = ['0', '1']

def sgd_train(X_train, X_test, y_train, y_test):
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, y_train)
    y_pred = sgd.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    # print('Accuracy: %s' % accuracy)  # test_size=0.1, random_state=42 - Multinomial Naive Bayes model
    report = classification_report(y_test, y_pred, target_names=tags)
    # print(report)
    return sgd, accuracy, report


#
#
# from pickle import load, dump
#
# dump(sgd, open("model.pkl", "wb"))
# model = load(open("model.pkl", "rb"))
#
# y_pred = model.predict(X_test)

# print('Accuracy %s' % accuracy_score(y_pred, y_test))
#
# print(classification_report(y_test, y_pred, target_names=tags))
#
# y_pred

def train(save=True):
    df = pd.read_csv("data/sinhala-hate-speech-dataset-processed-2.csv", encoding='utf8')
    # df = pd.read_csv("../Riyana/Research/Text-Classification-Module-master/Dataset/cleaned data-2.csv", encoding='utf8')

    # # keep only label and Text_cleaned
    # df = df[['label', 'Text_cleaned']]
    # df = df.dropna()
    # df = df.reset_index(drop=True)
    # # rename to Class and Filtered_sentence
    # df = df.rename(columns={'label': 'Class', 'Text_cleaned': 'Filtered_sentence'})

    # print(df)

    class_to_num = {
        "Class": {
            'Hate': 0,
            'Not Hate': 1
        }
    }

    df = df.replace(class_to_num)
    print(df)

    # split into train/test sets
    sentences = df['Text_cleaned'].values.astype('U')
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.1, random_state=42)

    # y to string
    y = y.astype(str)

    # print df types
    print(df.dtypes)

    # print if there is value not in tags in y
    for i in y:
        if i not in tags:
            print(i)


    nb = nb_train(X_train, X_test, y_train, y_test)

    knn = knn_train(X_train, X_test, y_train, y_test)

    RF = RF_train(X_train, X_test, y_train, y_test)

    LR = LR_train(X_train, X_test, y_train, y_test)

    svc = svc_train(X_train, X_test, y_train, y_test)

    sgd = sgd_train(X_train, X_test, y_train, y_test)

    print("Naive Bayes Accuracy: ", nb[1])
    print("KNN Accuracy: ", knn[1])
    print("Random Forest Accuracy: ", RF[1])
    print("Logistic Regression Accuracy: ", LR[1])
    print("SVC Accuracy: ", svc[1])
    print("SGD Accuracy: ", sgd[1])

    best_model = None
    # pick best model and accuracy
    best_model_accuracy = max(nb[1], knn[1], RF[1], LR[1], svc[1], sgd[1])
    print("\nBest Model Accuracy: ", best_model_accuracy)
    # best model name
    if best_model_accuracy == nb[1]:
        print("Best Model: Naive Bayes")
        best_model = nb[0]
    elif best_model_accuracy == knn[1]:
        print("Best Model: KNN")
        best_model = knn[0]
    elif best_model_accuracy == RF[1]:
        print("Best Model: Random Forest")
        best_model = RF[0]
    elif best_model_accuracy == LR[1]:
        print("Best Model: Logistic Regression")
        best_model = LR[0]
    elif best_model_accuracy == svc[1]:
        print("Best Model: SVC")
        best_model = svc[0]
    elif best_model_accuracy == sgd[1]:
        print("Best Model: SGD")
        best_model = sgd[0]
    else:
        print("No Best Model")

    dump(best_model, open("model_binary.pkl", "wb"))
    print("Model Saved")

    print("Training Completed")
    print("Model location: ", os.getcwd() + "/model_binary.pkl")

    return best_model


def predict_from_best(text, model=None):
    print([text])
    if model is None:
        model = load(open("model_binary.pkl", "rb"))

    pred = model.predict([text])
    print(labels[pred[0] - 1])

    return labels[pred[0] - 1]


labels = ['Hate', 'Not Hate']


# text = "මහ බැංකුවේ ලයිට් දාලා බයියෝ චූන් කරලා අදට හරියටම අවුරුද්දයි"

# pred = best_model.predict([text])
# print(labels[pred[0] - 1])

#
# def method_name(model, y_test, y_pred):
#     import seaborn as sn
#     sn.heatmap(confusion_matrix(y_test, y_pred), annot=True)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.show()
#     g = plt.scatter(y_test, y_pred)
#     g.axes.set_yscale('log')
#     g.axes.set_xscale('log')
#     g.axes.set_xlabel('True Values ')
#     g.axes.set_ylabel('Predictions ')
#     g.axes.axis('equal')
#     g.axes.axis('square')
#     g = plt.plot(y_test - y_pred, marker='o', linestyle='')
#     import seaborn as sns
#     from sklearn.metrics import roc_curve, auc
#     def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6)):
#         y_score = clf.decision_function(X_test)
#
#         # structures
#         fpr = dict()
#         tpr = dict()
#         roc_auc = dict()
#
#         # calculate dummies once
#         y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
#         for i in range(n_classes):
#             fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
#             roc_auc[i] = auc(fpr[i], tpr[i])
#
#         # roc for each class
#         fig, ax = plt.subplots(figsize=figsize)
#         ax.plot([0, 1], [0, 1], 'k--')
#         ax.set_xlim([0.0, 1.0])
#         ax.set_ylim([0.0, 1.05])
#         ax.set_xlabel('False Positive Rate')
#         ax.set_ylabel('True Positive Rate')
#         ax.set_title('Receiver operating characteristic for multiclass classification')
#         for i in range(n_classes):
#             ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
#         ax.legend(loc="best")
#         ax.grid(alpha=.4)
#         sns.despine()
#         plt.show()
#
#     plot_multiclass_roc(model, X_test, y_test, n_classes=6, figsize=(16, 10))

# method_name()

#
#
# nb_train(X_train, X_test, y_train, y_test)
#
# knn_train(X_train, X_test, y_train, y_test)
#
# RF_train(X_train, X_test, y_train, y_test)
#
# LR_train(X_train, X_test, y_train, y_test)
#
# svc_train(X_train, X_test, y_train, y_test)

train()
