import sinling
import advertools
import translate
# import pyenchant
import nltk

from sinling import SinhalaTokenizer as tokenizer, SinhalaStemmer as stemmer, POSTagger, preprocess, word_joiner, word_splitter
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.probability import FreqDist
import advertools as adv
from pathlib import Path
import string

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn import linear_model
import pygtrie as trie
import codecs
import nltk

nltk.download('punkt')
import re
from sinling.config import RESOURCE_PATH
from sinling.core import Stemmer

from nltk.corpus import stopwords
from collections import Counter

from lib.SinhaleseVowelLetterFixer import SinhaleseVowelLetterFixer
from lib.TranslateToSinhala import EnglishToSinhalaTranslator
from lib.StemWords import stem_word
from lib.PosTagger import get_pos_tags

from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok
from nltk.tokenize.treebank import TreebankWordDetokenizer

stemmer = stemmer()



def stem_word(word: str) -> str:
    # word= translate_to_sinhala(word)
    """
    Stemming words
    :param word: word
    :return: stemmed word
    """
    if len(word) < 4:
        return word

    # remove 'ට'
    if word[-1] == 'ට':
        return word[:-1]

    # remove 'ම'
    if word[-1] == 'ම':
        return word[:-1]

    # remove 'ද'
    if word[-1] == 'ද':
        return word[:-1]

    # remove 'ටත්'
    if word[-3:] == 'ටත්':
        return word[:-3]

    # remove 'එක්'
    if word[-3:] == 'ෙක්':
        return word[:-3]

    # remove 'යේ'
    if word[-2:] == 'යේ':
        return word[:-2]

    # remove 'ගෙ' (instead of ගේ because this step comes after simplifying text)
    if word[-2:] == 'ගෙ':
        return word[:-2]

    # remove 'එ'
    if word[-1:] == 'ෙ':
        return word[:-1]

    # remove 'ක්'
    if word[-2:] == 'ක්':
        return word[:-2]

    # remove 'වත්'
    if word[-3:] == 'වත්':
        return word[:-3]

    word = stemmer.stem(word)
    word = word[0]

    # else
    return word


def filter_stop_words(sentences):
    stopwords_set = ["සහ", "සමග", "සමඟ", "අහා", "ආහ්", "ආ", "ඕහෝ", "අනේ", "අඳෝ", "අපොයි", "පෝ", "අයියෝ", "ආයි", "ඌයි",
                     "චී", "චිහ්", "චික්", "හෝ‍", "දෝ",
                     "දෝහෝ", "මෙන්", "සේ", "වැනි", "බඳු", "වන්", "අයුරු", "අයුරින්", "ලෙස", "වැඩි", "ශ්‍රී", "හා", "ය",
                     "නිසා", "නිසාවෙන්", "බවට", "බව", "බවෙන්", "නම්", "වැඩි", "සිට",
                     "දී", "මහා", "මහ", "පමණ", "පමණින්", "පමන", "වන", "විට", "විටින්", "මේ", "මෙලෙස", "මෙයින්", "ඇති",
                     "ලෙස", "සිදු", "වශයෙන්", "යන", "සඳහා", "මගින්", "හෝ‍",
                     "ඉතා", "ඒ", "එම", "ද", "අතර", "විසින්", "සමග", "පිළිබඳව", "පිළිබඳ", "තුළ", "බව", "වැනි", "මහ",
                     "මෙම", "මෙහි", "මේ", "වෙත", "වෙතින්", "වෙතට", "වෙනුවෙන්",
                     "වෙනුවට", "වෙන", "ගැන", "නෑ", "අනුව", "නව", "පිළිබඳ", "විශේෂ", "දැනට", "එහෙන්", "මෙහෙන්", "එහේ",
                     "මෙහේ", "ම", "තවත්", "තව", "සහ", "දක්වා", "ට", "ගේ",
                     "එ", "ක", "ක්", "බවත්", "බවද", "මත", "ඇතුලු", "ඇතුළු", "මෙසේ", "වඩා", "වඩාත්ම", "නිති", "නිතිත්",
                     "නිතොර", "නිතර", "ඉක්බිති", "දැන්", "යලි", "පුන", "ඉතින්",
                     "සිට", "සිටන්", "පටන්", "තෙක්", "දක්වා", "සා", "තාක්", "තුවක්", "පවා", "ද", "හෝ‍", "වත්", "විනා",
                     "හැර", "මිස", "මුත්", "කිම", "කිම්", "ඇයි", "මන්ද", "හෙවත්",
                     "නොහොත්", "පතා", "පාසා", "ගානෙ", "තව", "ඉතා", "බොහෝ", "වහා", "සෙද", "සැනින්", "හනික", "එම්බා",
                     "එම්බල", "බොල", "නම්", "වනාහි", "කලී", "ඉඳුරා",
                     "අන්න", "ඔන්න", "මෙන්න", "උදෙසා", "පිණිස", "සඳහා", "රබයා", "නිසා", "එනිසා", "එබැවින්", "බැවින්",
                     "හෙයින්", "සේක්", "සේක", "ගැන", "අනුව", "පරිදි", "විට",
                     "තෙක්", "මෙතෙක්", "මේතාක්", "තුරු", "තුරා", "තුරාවට", "තුලින්", "නමුත්", "එනමුත්", "වස්", 'මෙන්',
                     "ලෙස", "පරිදි", "එහෙත්"]

    filtered_sentences = []
    detokenizer = Detok()
    for sentence in sentences:
        tokenized_sentence = word_tokenize(sentence)
        filtered_sentence = [word for word in tokenized_sentence if word not in stopwords_set]
        filtered_sentence = []
        for w in tokenized_sentence:
            if w not in stopwords_set:
                filtered_sentence.append(stem_word(w))
        filtered_sentences.append(filtered_sentence)
    return filtered_sentences


def Detokenioze(text):
    detokenized_sentences = []

    for sentence in text:
        detokenized_sentences.append(TreebankWordDetokenizer().detokenize(sentence))
    return detokenized_sentences


def list_to_dataframe(list):
    df = pd.DataFrame(list)
    return df


def dataframe_to_list(dataframe):
    list = dataframe.values.tolist()
    return list


def clean_data(dataframe):
    # Drop duplicate rows
    dataframe.drop_duplicates(subset='Text', inplace=True)
    # replace URL of a text
    dataframe['Text_cleaned'] = dataframe['Text'].str.replace(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True)
    # replace mention
    dataframe['Text_cleaned'] = dataframe['Text_cleaned'].str.replace('#|@\w*', '', regex=True)
    # remove retweet states in the beginning such as "RT @sam92ky: "
    dataframe['Text_cleaned'] = dataframe['Text_cleaned'].str.replace('RT : ', '')
    # remove numbers
    dataframe['Text_cleaned'] = dataframe['Text_cleaned'].str.replace('\d+', '', regex=True)

    # punctuation removal
    string_text = dataframe['Text_cleaned'].str
    dataframe['Text_cleaned'] = string_text.translate(str.maketrans('', '', string.punctuation))
    print(string_text)

    # coerced entire coloumn to str dtype
    dataframe['Text_cleaned'] = dataframe['Text_cleaned'].astype(str)

    # translate English to sinhala
    # df['Text_cleaned'] = df['Text_cleaned'].apply(translate_to_sinhala).tolist()
    # df['Text_cleaned'] = df['Text_cleaned'].apply(translate_english).tolist()
    #
    # # simplify sinhala characters
    # df['Text_cleaned'] = df['Text_cleaned'].apply(simplify_sinhalese_text).tolist()

    # pos tagging
    # df['Text'] = df['Text'].apply(tagger.predict).tolist()

    # print("New shape:", dataframe.shape)
    return dataframe['Text_cleaned']


######## Simplify sinhalese text ########

# dictionary that maps wrong usage of vowels to correct vowels
simplify_characters_dict = {
    # Consonant
    "ඛ": "ක",
    "ඝ": "ග",
    "ඟ": "ග",
    "ඡ": "ච",
    "ඣ": "ජ",
    "ඦ": "ජ",
    "ඤ": "ඥ",
    "ඨ": "ට",
    "ඪ": "ඩ",
    "ණ": "න",
    "ඳ": "ද",
    "ඵ": "ප",
    "භ": "බ",
    "ඹ": "බ",
    "ශ": "ෂ",
    "ළ": "ල",

    # Vowels
    "ආ": "අ",
    "ඈ": "ඇ",
    "ඊ": "ඉ",
    "ඌ": "උ",
    "ඒ": "එ",
    "ඕ": "ඔ",

}

def get_simplified_character(character: str) -> str:
    if len(character) != 1:
        raise TypeError("character should be a string with length 1")
    try:
        return simplify_characters_dict[character]
    except KeyError:
        return character


def simplify_sinhalese_text(text: str) -> str:
    """
    simplify
    :param text:
    :return:
    """
    modified_text = ""
    for c in text:
        modified_text += get_simplified_character(c)
    return modified_text






file_path = '../recognized.txt'


# read text file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def process_data(file_path):
    print(read_text_file(file_path))

    # split text into sentences
    sentences = sent_tokenize(read_text_file(file_path))
    print(sentences, '\n')

    # remove stopwords
    filtered_sentences = filter_stop_words(sentences)
    print(filtered_sentences, '\n')

    # detokenize
    detokenized_sentences = Detokenioze(filtered_sentences)
    print(detokenized_sentences, '\n')

    # simplify
    simplified_sentences = []
    for sentence in detokenized_sentences:
        simplified_sentences.append(simplify_sinhalese_text(sentence))
    print(simplified_sentences, '\n')

    # clean data
    df = pd.DataFrame(simplified_sentences, columns=['Text'])
    df = clean_data(df)
    cleaned_sentences = dataframe_to_list(df)
    print(cleaned_sentences, '\n')

    df = pd.DataFrame(simplified_sentences, columns=['Text'])
    df = clean_data(df)

    # save to csv
    # df.to_csv('cleaned_data.csv', index=False)

    fixed_sentences = []
    for sentence in cleaned_sentences:
        # vowel letter fixer
        fixed_sentence = SinhaleseVowelLetterFixer.get_fixed_text(sentence)
        fixed_sentences.append(fixed_sentence)
    print(fixed_sentences, '\n')

    # Translate to sinhala
    translated_sentences = []
    for sentence in fixed_sentences:
        translated_sentences.append(EnglishToSinhalaTranslator.translate_to_sinhala(sentence))
    print("Translated sentences: ", translated_sentences, '\n')

    # POS tagging
    # tagged_sentences = []
    # for sentence in translated_sentences:
    #     tagged_sentences.append(get_pos_tags(text=sentence))
    # print(tagged_sentences, '\n')

    # Translate to english
    english_removed = []
    for sentence in translated_sentences:
        english_removed.append(EnglishToSinhalaTranslator.remove_english_words(sentence))
    print("English removed: ", english_removed, '\n')

    return english_removed