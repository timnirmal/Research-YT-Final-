import nltk
import pandas as pd
import re
import string
import os

# print current working directory
print("Current working directory: ", os.getcwd())

# english-sinhala dictionary
dictionary = {}
df = pd.read_csv('data_files/en-sinhala dictionary.csv')
dictionary_file = df["En,sinhala"]



for line in dictionary_file:
    key, value = line.strip().split(",")
    dictionary[key] = value

translate_words_dict = {
    "unp": "එක්සත් ජාතික පක්ෂය",
    "muslim": "මුස්ලිම්",
    "srilankanpolitics": "ශ්‍රී ලංකන් දේශපාලනය",
    "council": "සභාව",
    "sinhala": "සිංහල",
    "buddhist": "බෞද්ධ",
    "buddhism": "බුද්ධාගම",
    "srilanka": "ශ්‍රී ලංකාව",
    "racist": "ජාතිවාදී",
    "presidentialfirst": "පළමු ජනාධිපති",
    "feeling": "හැඟීම",
    "feminist": "ස්ත්‍රීවාදී",
    "loved": "ආදරය කළා",
    "team": "කණ්ඩායම",
    "tclsl": "ට්විටර් ක්‍රිකට් ලීගය ශ්‍රී ලංකාව",
    "pongal": "පොංගල්",
    "pongalfestival": "පොංගල් උත්සවය",
    "women": "කාන්තා",
    "nextpresidentinsl": "ශ්‍රී ලංකාවේ මීළඟ ජනාධිපති ",
    "seventhexecutivepresident": "හත්වන විධායක සභාපති",
    "hate": "වෛරය",
    "love": "ආදරය",
    "angry": "තරහයි",
    "doctor": "ඩොක්ටර්",
    "ltte": "එල්ටීටීඊය",
    "lka": "‍ශ්‍රී ලංකාව",
    "hurt": "රිදෙනවා",
    "typo": "යතුරු ලියනය",
    "racial": "වාර්ගික",
    "hatred": "වෛරය",
    "halal": "හලාල්",
    "wicket": "කඩුල්ල",
    "taker": "ටේකර්",
    "indoor": "ගෘහස්ථ",
    "attacker": "ප්‍රහාරකයා",
    "attack": "ප්රහාරය",
    "spikers": "ස්පිකර්ස්",
    "training": "පුහුණුව",
    "final": "අවසාන",
    "match": "තරගය",
    "tournament": "තරඟාවලිය",
    "youth": "තරුණ",
    "amen": "ආමෙන්",
    "enough": "ඇති",
    "standagainstracism": "ජාතිවාදයට එරෙහිව නැගී සිටින්න"
}


# to check whether the string contains English words(any)
def translate_english(x):
    for word1 in x.split():
        new_word = ''.join(i for i in word1 if not i.isdigit())
        x = x.replace(word1, new_word)
    for word in x.split():
        word2 = "".join(l for l in word if l not in string.punctuation)
        if re.match('[a-zA-Z]', word2) is not None:
            word1 = word2.lower()
            translated_word = dictionary.get(word1)
            if translated_word is None:
                translated_word = ''
            x = x.replace(word, translated_word)
    return x


class EnglishToSinhalaTranslator:
    """
    English to Sinhala translator
    """

    def translate_to_sinhala(word: str) -> str:
        word = word.lower()
        if word in translate_words_dict:
            return translate_words_dict[word]
        return word

    # translate_to_sinhala("unp")

    import nltk
    nltk.download('words')
    import string

    def remove_english_words(sent: str) -> str:
        words = set(nltk.corpus.words.words())
        string_text = "".join(w for w in sent if not w in words)

        string_text = string_text.translate(str.maketrans('', '', string.punctuation))

        return string_text

