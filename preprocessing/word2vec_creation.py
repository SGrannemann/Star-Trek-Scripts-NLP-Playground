"""Script to create the word2vec and doc2vec models with the optimal hyperparameters."""
import multiprocessing
from pathlib import Path
from gensim.models import Word2Vec


# train the word2vec model
word2vec_model = Word2Vec(corpus_file=str(Path.cwd() / Path('star_trek_corpus.txt')),
                          workers=multiprocessing.cpu_count(), sg=1, min_count=1)

word2vec_model.save('word2vec.model')

# save the models vectors (which are keyed vectors in gensim) as we do not
# need to train the model further
word2vec_model.wv.save('wordvectors.kv')

model = Word2Vec.load('word2vec.model')


print(model.wv.most_similar(positive=['Picard']))
