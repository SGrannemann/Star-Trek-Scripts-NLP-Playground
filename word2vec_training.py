from gensim.models import Word2Vec
from pathlib import Path

word2vec_model = Word2Vec(corpus_file=str(Path.cwd() / Path('data')) + '/star_trek_corpus.txt',  size=300, sg=1, hs=0, negative=5, min_count=1)
word2vec_model.save('word2vec.model')

#model = Word2Vec.load('word2vec.model')

#print(model.wv.most_similar(positive=['41153.7']))
