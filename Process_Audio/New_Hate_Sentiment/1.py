import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
import csv
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# read csv data
data = pd.read_csv("Sinhala_Singlish_Hate_Speech.csv")

### Data preprocessing

# preporocessing texts
def preporocessingText(sentence):
    # regex for html tags cleaner
    cleaner_htmlTags = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext_htmltags = re.sub(cleaner_htmlTags, '', sentence).lower()  # convert to lower case

    # regex for non alphabetical characters cleaner
    cleantext_NonAlp = re.compile(u'[^\u0061-\u007A|^\u0D80-\u0DFF|^\u0030-\u0039]', re.UNICODE)
    # Englosh lower case unicode range = \u0061-\u007A
    # Sinhala unicode range = |u0D80-\u0DFF
    # Numbers unicode range = \u0030-\u0039

    cleantext_finalText = re.sub(cleantext_NonAlp, ' ', cleantext_htmltags).strip(" ")

    # tokenzing
    # finalText = word_tokenize(cleantext_finalText)
    # finalText = sent_token = nltk.sent_tokenize(tokenzie_finalText)

    # return finalText
    return cleantext_finalText

def isSinglish(sentence):
    try:
        sentence.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


# genarate separate dataframes
df_Sinhala = pd.DataFrame(columns=['PhraseNo', 'Phrase', 'IsHateSpeech'])
df_Singlish = pd.DataFrame(columns=['PhraseNo', 'Phrase', 'IsHateSpeech'])
df_All_preprocess = pd.DataFrame(columns=['PhraseNo', 'Phrase', 'IsHateSpeech'])

singish_index = 1
sinhala_index = 1

for i in range(data.shape[0]):
    dataSentence = data['Phrase'][i]
    preprocessData = preporocessingText(dataSentence)

    if (isSinglish(dataSentence)):
        df_Singlish.loc[singish_index - 1] = [singish_index] + [preprocessData] + [data['IsHateSpeech'][i]]
        singish_index += 1

    else:
        df_Sinhala.loc[sinhala_index - 1] = [sinhala_index] + [preprocessData] + [data['IsHateSpeech'][i]]
        sinhala_index += 1
    df_All_preprocess.loc[i] = [i + 1] + [preprocessData] + [data['IsHateSpeech'][i]]

# df_Sinhala.to_csv('Sinhala_hatespeech.csv')
# df_Singlish.to_csv('Singlish_hatespeech.csv')
# df_All_preprocess.to_csv('AllPreProcess_hatespeech.csv')


sinhala_sent_percentage = (sinhala_index/data.shape[0])*100
singlish_sent_percentage = (singish_index/data.shape[0])*100
print("Sinhala sentences percentage(mix sinhala and english letters)  = ",str(sinhala_sent_percentage)+" %")
print("Singlish sentences percentage(only has english letters) = ",str(singlish_sent_percentage) +" %")

def getUniqueTokens(myarr):
    myset = list(set(myarr))
    return myset

def joinWordsIntoSentence(dataframe):
    for itm in range(len(dataframe)):
        words_arr = dataframe['Phrase'][itm]
        dataframe['Phrase'][itm] = ( " ".join( words_arr ))



### For Sinhala Dataset(has sinhala end english words)
#getting the stop words
f_stopWords = io.open("StopWords_425.txt", mode="r", encoding="utf-16")
sinhala_stop_words = []
df_StopWordsRemoval_Sinhala = pd.DataFrame(columns=['PhraseNo', 'Phrase', 'IsHateSpeech'])

for x in f_stopWords:
  sinhala_stop_words.append(x.split()[0])

SinhalaData = df_Sinhala
prev_lengths_arr = []
prev_lengths_arr_unique = []
after_removal_stopWords_lenghts_arr = []
after_removal_stopWords_lenghts_arr_unique = []

for k in range(SinhalaData.shape[0]):
    SentenceTokens = word_tokenize(SinhalaData['Phrase'][k])

    prev_lengths_arr.append(len(SentenceTokens))
    # print(len(SentenceTokens))
    prev_lengths_arr_unique.append(len(getUniqueTokens(SentenceTokens)))
    # remove stop words
    removing_stopwords_sentence = [word for word in SentenceTokens if word not in sinhala_stop_words]
    after_removal_stopWords_lenghts_arr.append(len(removing_stopwords_sentence))
    after_removal_stopWords_lenghts_arr_unique.append(len(getUniqueTokens(removing_stopwords_sentence)))
    # print(removing_stopwords_sentence)
    df_StopWordsRemoval_Sinhala.loc[k] = [k + 1] + [removing_stopwords_sentence] + [SinhalaData['IsHateSpeech'][k]]

joinWordsIntoSentence(df_StopWordsRemoval_Sinhala)

print(df_StopWordsRemoval_Sinhala.head(10))

preLenghts = prev_lengths_arr
afterLenghts = after_removal_stopWords_lenghts_arr
preLenghtsUnique = prev_lengths_arr_unique
afterLenghtsUnique = after_removal_stopWords_lenghts_arr_unique

arr1 = []
arr2 = []
for i in range(len(preLenghts)):
    ar_i = []
    ar_i.append(preLenghts[i])
    ar_i.append(afterLenghts[i])
    arr1.append(ar_i)

    ar_j = []
    ar_j.append(preLenghtsUnique[i])
    ar_j.append(afterLenghtsUnique[i])
    arr2.append(ar_j)

import numpy as np

myarray1 = np.asarray(arr1)
myarray2 = np.asarray(arr2)

import matplotlib.pyplot as plt

np.random.seed(19680801)

n_bins = 30
x = np.random.randn(1000, 3)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 7))
ax0, ax1 = axes.flatten()

colors = ['red', 'tan']

ax0.hist(myarray1, n_bins, density=True, histtype='bar', stacked=True, label=colors)
ax0.set_title('sentences(before and after stopwordsremoval)')

ax1.hist(myarray2, n_bins, density=True, histtype='bar', stacked=True, color=colors, label=colors)
ax1.set_title('unique sentences(before and after stopwordsremoval)')

fig.tight_layout()
plt.show()


print("Max length(previous) = ",max(prev_lengths_arr))
print("Max length(after removal of stopwords) = ",max(after_removal_stopWords_lenghts_arr))
print()
print("Min length(previous) = ",min(prev_lengths_arr))
print("Min length(after removal of stopwords) = ",min(after_removal_stopWords_lenghts_arr))


f_suffixes = io.open("Suffixes-413.txt", mode="r", encoding="utf-16")
sinhala_suffixes = []
df_Stemming_Sinhala = pd.DataFrame(columns=['PhraseNo', 'Phrase', 'IsHateSpeech'])
afterStemingLenghtsUnique = []

for suf in f_suffixes:
  sinhala_suffixes.append(suf.strip().split()[0])


def isSuffix(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    if (n1 > n2):
        return False
    for i in range(n1):
        if (s1[n1 - i - 1] != s2[n2 - i - 1]):
            return False
    return True


def removeSuffix(word, suffix):
    newLen = len(word) - len(suffix)
    wordN = word[0:newLen]
    return wordN


def stemming(data_frame):
    stems = {}
    found = 0
    df_Stemming = pd.DataFrame(columns=['PhraseNo', 'Phrase', 'IsHateSpeech'])
    for r in range(data_frame.shape[0]):
        Sentence = data_frame['Phrase'][r]
        # print(Sentence)
        SentenceTokens = word_tokenize(Sentence)
        stemming_sentence_n = []
        for wr in SentenceTokens:
            found = 0
            for suf in sinhala_suffixes:
                if (isSuffix(suf.strip(), wr.strip())):
                    stm = removeSuffix(wr.strip(), suf.strip())
                    stems[wr] = stm
                    stemming_sentence_n.append(stems[wr])
                    found = 1
                    break

            if (found == 0):
                stemming_sentence_n.append(wr)

        # print(stemming_sentence_n)
        df_Stemming.loc[r] = [r + 1] + [stemming_sentence_n] + [data_frame['IsHateSpeech'][r]]
        stemming_sentence_n = []
        # print(stemming_sentence_n)
    print(stems)
    joinWordsIntoSentence(df_Stemming)
    return df_Stemming



df_Stemming_Sinhala = stemming(df_StopWordsRemoval_Sinhala)

print(df_Stemming_Sinhala.head(10))



### Bag of Words approach


from sklearn.model_selection import train_test_split

X = df_Stemming_Sinhala["Phrase"]
y = df_Stemming_Sinhala["IsHateSpeech"]
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.2, random_state = 0)


def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer

    print('Creating bag of words...')
    # Initialize the "CountVectorizer" object

    # In this example features may be single words or two consecutive words
    vectorizer = CountVectorizer()

    train_data_features = vectorizer.fit_transform(X)

    # Convert to a NumPy array for easy of handling
    train_data_features = train_data_features.toarray()

    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print('Created')
    return vectorizer, vocab, train_data_features

vectorizer, vocab, train_data_features  = (
        create_bag_of_words(X_train))

print(vocab)

def train_logistic_regression(features, label):
    print ("Training model........")
    from sklearn.linear_model import LogisticRegression
    #ml_model = LogisticRegression(C = 100,random_state = 0)
    ml_model = LogisticRegression()
    ml_model.fit(features, label)
    print ('Created')
    return ml_model





