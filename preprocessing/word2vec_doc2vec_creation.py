"""Script to create the word2vec and doc2vec documents with the optimal hyperparameters."""
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pathlib import Path
from pathlib import Path
import pandas as pd
from nltk.tokenize import sent_tokenize
import multiprocessing


# train the word2vec model

scripts_and_plots = pd.read_pickle(str(Path.cwd() / Path('data')) + '/scripts_and_plots.pkl')
complete_corpus = list(scripts_and_plots['complete_text_with_bigrams'])
episodes_prepared = [' '.join(sent_tokenize(episode)) for episode in scripts_and_plots['complete_text_with_bigrams']]

word2vec_model = Word2Vec(corpus_file=str(Path.cwd() / Path('data')) + '/star_trek_corpus.txt', min_count=1, iter=20, size=150, sg=1, hs=0, workers=multiprocessing.cpu_count(), negative=10, sample=0)

word2vec_model.save('word2vec.model')
# save the models vectors (which are keyed vectors in gensim) as we do not
# need to train the model further
word2vec_model.wv.save('wordvectors.kv')

model = Word2Vec.load('word2vec.model')




# train the doc2vec model

# create tagged documents for doc2vec, we do not need special categories.
DOCUMENTS = [TaggedDocument(doc, [i]) for i, doc in enumerate(episodes_prepared)]

doc2vec_model = Doc2Vec(documents=DOCUMENTS, dm=1, workers=3, hs=0, epochs=50, min_count=5, vector_size=150, negative=15, sample=0)

doc2vec_model.save('doc2vec.model')

#print(model.wv.most_similar(positive=['41153.7']))
