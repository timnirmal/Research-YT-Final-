import os

import pygtrie as trie
from sinling import SinhalaStemmer as stemmer
from sinling.config import RESOURCE_PATH
from sinling.core import Stemmer

__all__ = ['SinhalaStemmer']


def _load_stem_dictionary():
    stem_dict = dict()
    with open(os.path.join(RESOURCE_PATH, 'stem_dictionary.txt'), 'r', encoding='utf-8') as fp:
        for line in fp.read().split('\n'):
            try:
                base, suffix = line.strip().split('\t')
                stem_dict[f'{base}{suffix}'] = (base, suffix)
            except ValueError as _:
                pass
    return stem_dict


def _load_suffixes():
    suffixes = trie.Trie()
    with open(os.path.join(RESOURCE_PATH, 'suffixes_list.txt'), 'r', encoding='utf-8') as fp:
        for suffix in fp.read().split('\n'):
            suffixes[suffix[::-1]] = suffix
    return suffixes


class SinhalaStemmer(Stemmer):
    def __init__(self):
        super().__init__()
        self.stem_dictionary = _load_stem_dictionary()
        self.suffixes = _load_suffixes()

    def stem(self, word):
        if word in self.stem_dictionary:
            return self.stem_dictionary[word]
        else:
            suffix = self.suffixes.longest_prefix(word[::-1]).key
            if suffix is not None:
                return word[0:-len(suffix)], word[len(word) - len(suffix):]
            else:
                return word, ''


stemmer = stemmer()


def stem_word(word: str) -> str:
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

# print(stem_word('ගරසරප'))
# print(stem_word('චිත්‍රපටය'))
# print(stem_word('පහල'))
# print(stem_word('තියෙන'))
# print(stem_word('සම්බන්ධකය'))
# print(stem_word('එකෙන්'))
# print(stem_word('බාගත'))
# print(stem_word('කරගන්න'))
# විපක්ෂයේ,විපක්ෂව,විපක්ෂකම,විපක්ෂය,විපක්ෂයා

# print(stem_word('ගරසරප චිත්‍රපටය පහල තියෙන සම්බන්ධකය එකෙන් බාගත කරගන්න'))

# words = ["නායකයා","නායකයෝ","නායකයන්","නායකයාට","නායකයන්ට","නායකයාගෙන්","නායකයන්ගෙන්","නායකයාගේ","නායකයන්ගේ","නායකයනි"]
# words = ["ගරසරප", "චිත්‍රපටය", "පහල", "තියෙන", "සම්බන්ධකය", "එකෙන්", "බාගත", "කරගන්න"]
