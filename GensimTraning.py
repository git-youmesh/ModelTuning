"""sentences: a list of lists of tokens, the data to train the model on
min_count: the minimum frequency of a word to be considered for embedding (the default value is 5, and so below we will manually set this to 1 to force the model to encode every word)
vector_size: the number of embedding dimensions
window: window size when training the model
sg: a binary value where 0 indicates a CBOW model should be used (which the default), and 1 indicates Skip-Gram should be used.
"""

import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import gensim.downloader
# Print out the datasets
corpora = gensim.downloader.info()['corpora'].keys()
for corpus in corpora:
    print(corpus)

# Print out the pre-trained models
models = gensim.downloader.info()['models'].keys()
for model in models:
    print(model)




# Create a model by loading the text8 dataset
corpus = api.load('text8')

# Create a CBOW model
"""cbow_model = Word2Vec(corpus,
                      min_count=1,
                      vector_size=5,
                      window=4)

# Create a Skip-Gram model
skipgram_model = Word2Vec(corpus,
                          min_count=1,
                          vector_size=5,
                          window=4,
                          sg=True)"""
# Print the model description
model_dict = gensim.downloader.info()['models']['word2vec-google-news-300']

for key in ['num_records', 'base_dataset', 'description']:
    print(f'{key: <12}: {model_dict[key]}')

# Download in the model
google_cbow = api.load('word2vec-google-news-300')


# Return the embedding for a word
#print('Word Embedding for "tree":\n')
#print(f'CBOW:        {cbow_model.wv["tree"]}')
#print(f'Skip-Gram:   {skipgram_model.wv["tree"]}')
print(f'Google CBOW: {google_cbow["tree"][:5]}\n\n')


# Calculate the similarity between words
print('Similarity Between "tree" and "leaf":\n')
#print(f'CBOW:        {cbow_model.wv.similarity("tree", "leaf")}')
#print(f'Skip-Gram:   {skipgram_model.wv.similarity("tree", "leaf")}')
print(f'Google CBOW: {google_cbow.similarity("tree", "leaf")}\n\n')


# Return the top 3 most similiar words
print('Most Similar Words to "tree":\n')
#print(f'CBOW:        {cbow_model.wv.most_similar("tree", topn=3)}')
#print(f'Skip-Gram:   {skipgram_model.wv.most_similar("tree", topn=3)}')
print(f'Google CBOW: {google_cbow.most_similar("tree", topn=3)}\n\n')


# Find which word doesn't match
words = ['tree', 'leaf', 'plant', 'bark', 'car']

#cbow_result = cbow_model.wv.doesnt_match(words)
#skipgram_result = skipgram_model.wv.doesnt_match(words)
google_result = google_cbow.doesnt_match(words)

print(f"Find Which Word Doesn't Match: {words}:\n")
#print(f'CBOW:        {cbow_result}')
#print(f'Skip-Gram:   {skipgram_result}')
print(f'Google CBOW: {google_result}')


