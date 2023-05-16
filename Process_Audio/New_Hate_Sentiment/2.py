import pandas as pd


data_path = r"D:\Projects\Research-YT\Research-YT-Final\Process_Audio\New_Hate_Sentiment\Sinhala_Singlish_Hate_Speech_Processed.csv"

data = pd.read_csv(data_path)

#Now lets split the data
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, shuffle = True, stratify = data.label, train_size = 50/data.shape[0], random_state = 50)
test, _ = train_test_split(test, shuffle = True, stratify = test.label, train_size = 10000/test.shape[0], random_state = 50)

print(train.shape, test.shape)

# Adverserial Validation
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold


def adversarial_validation(X, Y, n_splits=10):
    # Combine both datasets
    sparse_merge = sparse.vstack((X, Y))

    # Label the datasets
    y = np.array([0 for _ in range(X.shape[0])] + [1 for _ in range(Y.shape[0])])

    # Do 10 Fold CV
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    lr_auc = np.array([])
    rf_auc = np.array([])
    for train_idx, test_idx in kfold.split(sparse_merge, y):
        # Run Log Reg
        x_train, y_train = sparse_merge[train_idx], y[train_idx]
        x_test, y_test = sparse_merge[test_idx], y[test_idx]

        log_reg = SGDClassifier(loss='log')
        log_reg.fit(x_train, y_train)
        y_test_prob = log_reg.predict_proba(x_test)[:, 1]
        lr_auc = np.append(lr_auc, roc_auc_score(y_test, y_test_prob))
        # Run RF
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf.fit(x_train, y_train)
        y_test_prob = rf.predict_proba(x_test)[:, 1]
        rf_auc = np.append(rf_auc, roc_auc_score(y_test, y_test_prob))

    # Display results
    print('Logisitic Regression AUC : {:.3f}'.format(lr_auc.mean()))
    print('Random Forest AUC : {:.3f}'.format(rf_auc.mean()))

bow = CountVectorizer()
x_train = bow.fit_transform(train.title.values)
x_test = bow.transform(test.title.values)
x_test = shuffle(x_test)
adversarial_validation(x_train, x_test[:50])

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score, roc_auc_score, confusion_matrix
import seaborn as sns

sns.set_palette("muted")


def calc_f1(p_and_r):
    p, r = p_and_r
    return (2 * p * r) / (p + r)


# Print the F1, Precision, Recall, ROC-AUC, and Accuracy Metrics
# Since we are optimizing for F1 score - we will first calculate precision and recall and
# then find the probability threshold value that gives us the best F1 score

def print_model_metrics(y_test, y_test_prob, confusion=False, verbose=True, return_metrics=False):
    precision, recall, threshold = precision_recall_curve(y_test, y_test_prob, pos_label=1)

    # Find the threshold value that gives the best F1 Score
    best_f1_index = np.argmax([calc_f1(p_r) for p_r in zip(precision, recall)])
    best_threshold, best_precision, best_recall = threshold[best_f1_index], precision[best_f1_index], recall[
        best_f1_index]

    # Calulcate predictions based on the threshold value
    y_test_pred = np.where(y_test_prob > best_threshold, 1, 0)

    # Calculate all metrics
    f1 = f1_score(y_test, y_test_pred, pos_label=1, average='binary')
    roc_auc = roc_auc_score(y_test, y_test_prob)
    acc = accuracy_score(y_test, y_test_pred)

    if confusion:
        # Calculate and Display the confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred)

        plt.title('Confusion Matrix')
        sns.set(font_scale=1.0)  # for label size
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=['No Clickbait', 'Clickbait'],
                    yticklabels=['No Clickbait', 'Clickbait'], annot_kws={"size": 14}, cmap='Blues')  # font size

        plt.xlabel('Truth')
        plt.ylabel('Prediction')

    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(f1, best_precision,
                                                                                                best_recall, roc_auc,
                                                                                                acc))

    if return_metrics:
        return np.array([f1, best_precision, best_recall, roc_auc, acc])


# Run Simple Log Reg Model and Print metrics
from sklearn.linear_model import SGDClassifier


# Run log reg 10 times and average the result to reduce predction variance
def run_log_reg(train_features, test_features, y_train, y_test, alpha=1e-4, confusion=False, return_f1=False,
                verbose=True):
    metrics = np.zeros(5)
    for _ in range(10):
        log_reg = SGDClassifier(loss='log', alpha=alpha, n_jobs=-1, penalty='l2')
        log_reg.fit(train_features, y_train)
        y_test_prob = log_reg.predict_proba(test_features)[:, 1]
        metrics += print_model_metrics(y_test, y_test_prob, confusion=confusion, verbose=False, return_metrics=True)
    metrics /= 10
    if verbose:
        print('F1: {:.3f} | Pr: {:.3f} | Re: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f} \n'.format(*metrics))
    if return_f1:
        return f1
    return log_reg


y_train = np.where(train.label.values == 'clickbait', 1, 0)
y_test = np.where(test.label.values == 'clickbait', 1, 0)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
bow = CountVectorizer()
x_train = bow.fit_transform(train.title.values)
x_test = bow.transform(test.title.values)
run_log_reg(x_train, x_test, y_train, y_test)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
x_train = tfidf.fit_transform(train.title.values)
x_test = tfidf.transform(test.title.values)
run_log_reg(x_train, x_test, y_train, y_test)


# We'll use Average Glove here
from tqdm import tqdm_notebook
from nltk import word_tokenize
from pymagnitude import *
glove = Magnitude("./vectors/glove.6B.100d.magnitude")
def avg_glove(df):
    vectors = []
    for title in tqdm_notebook(df.title.values):
        vectors.append(np.average(glove.query(word_tokenize(title)), axis = 0))
    return np.array(vectors)
x_train = avg_glove(train)
x_test = avg_glove(test)
run_log_reg(x_train, x_test, y_train, y_test)

from InferSent.models import InferSent
import torch
MODEL_PATH = './encoder/infersent1.pkl'
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,'pool_type': 'max', 'dpout_model': 0.0, 'version': 1}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
infersent.set_w2v_path('GloVe/glove.840B.300d.txt')
infersent.build_vocab(train.title.values, tokenize= False)
x_train = infersent.encode(train.title.values, tokenize= False)
x_test = infersent.encode(test.title.values, tokenize= False)
run_log_reg(x_train, x_test, y_train, y_test, alpha = 1e-4)


from featurization import *
train_features, test_features, feature_names = featurize(train, test, 'tfidf_glove')
run_log_reg(train_features, test_features, y_train, y_test, alpha = 5e-2)

from sklearn.linear_model import SGDClassifier
import eli5
# Train a Log Reg Classifier
log_reg = SGDClassifier(loss = 'log', n_jobs = -1, alpha = 5e-2)
log_reg.fit(train_features, y_train)
#Pass the model instance along with the feature names to ELI5
eli5.show_weights(log_reg, feature_names = feature_names, top = 100)





