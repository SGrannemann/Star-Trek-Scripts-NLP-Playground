import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize


scripts_and_plots = pd.read_pickle(str(Path.cwd() / Path('data')) + '/scripts_and_plots.pkl')
complete_corpus = list(scripts_and_plots['complete_text_with_bigrams'])





# create the corpus for word2vec

sentences = []
tokens = []
for episode in complete_corpus:
    for sentence in sent_tokenize(episode):
        sentences.append(sentence)


        
with open(str(Path.cwd() / Path('data')) + '/star_trek_corpus.txt', 'w', encoding='utf-8') as corpus_file:
    for sentence in sentences:
        corpus_file.write(sentence + '\n')


# create the corpus for doc2vec
episodes = []

for episode in complete_corpus:
    episodes.append(' '.join(sent_tokenize(episode)) + '\n')

with open(str(Path.cwd() / Path('data')) + '/star_trek_corpus_doc2vec.txt', 'w', encoding='utf-8') as corpus_file:
    corpus_file.writelines(episodes)



