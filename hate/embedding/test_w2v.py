import gensim
from gensim.models import FastText, word2vec

model = "word2vec_300.w2v"
fasttext = "fasttext_300.w2v"

# Load the pre-trained w2v file
w2v_model = word2vec.Word2Vec.load("word2vec_300.w2v")
# fasttext_model = FastText.load("fasttext_300.w2v")

# Retrieve the vector for a given word
word = 'පැත්තක්'
vector = w2v_model.wv[word]
print(vector)

# Most similar words
similar = w2v_model.wv.most_similar(word)
print(similar)

# Similarity between two words
similarity = w2v_model.wv.similarity('පැත්තක්', 'ඇඟිල්ලක්')

print(similarity)

word = "ආසාධරණය"
print(word)
print(w2v_model.wv.most_similar(word))
vector = w2v_model.wv[word]
print(vector)


# # Odd one out
# odd = w2v_model.wv.doesnt_match(['පැත්තක්', 'පැත්තක්', 'පැත්තක්', 'පැත්තක්'])
# print(odd)
#
# # Vector arithmetic
# result = w2v_model.wv.most_similar(positive=['පැත්තක්', 'පැත්තක්'], negative=['පැත්තක්'])
# print(result)
#
# # # Load the pre-trained fasttext file
# # fasttext_model = FastText.load("fasttext_300.w2v")
#

