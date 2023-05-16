import os.path

import pandas as pd
from joblib import dump, load
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import *
from sklearn import ensemble
from sklearn import linear_model


# show all columns
pd.set_option('display.max_columns', None)

model_path = "model_files/hate_binary/"

tags = ['0', '1']


################################################## Multinomial Naive Bayes model

# NB
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
    df = pd.read_csv("Process_Audio/data/sinhala-hate-speech-dataset-processed-2.csv", encoding='utf8')
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

    dump(best_model, open(model_path + "model_binary.pkl", "wb"))
    print("Model Saved")

    print("Training Completed")
    print("Model location: ", model_path + "model_binary.pkl")

    return best_model


def predict_from_best(text, model=None):
    labels = ['Hate', 'Not Hate']

    print([text])
    if model is None:
        model = load(open(model_path + "model_binary.pkl", "rb"))

    pred = model.predict([text])
    # print(labels[pred[0] - 1])

    return labels[pred[0] - 1]