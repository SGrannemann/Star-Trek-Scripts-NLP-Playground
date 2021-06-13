from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from pathlib import Path
import pandas as pd
from nltk.tokenize import sent_tokenize

scripts_and_plots = pd.read_pickle(str(Path.cwd() / Path('data')) + '/scripts_and_plots.pkl')
complete_corpus = list(scripts_and_plots['complete_text_with_bigrams'])

episodes = []

for episode in complete_corpus:
    episodes.append(' '.join(sent_tokenize(episode)))


documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(episodes)]

doc2vec_model = Doc2Vec(documents=documents, vector_size=300, min_count=1, workers=3, hs=0, negative=10)

doc2vec_model.save('doc2vecmodel.model')