import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize
import spacy

scripts_and_plots = pd.read_pickle(str(Path.cwd() / Path('data')) + '/scripts_and_plots.pkl')
complete_corpus = list(scripts_and_plots['complete_text_with_bigrams'])

nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'lemmatizer'])



sentences = []
tokens = []
for episode in complete_corpus:
    for sentence in sent_tokenize(episode):
        sentences.append(sentence)
#for sentence in sentences:
#    tokens.append(sentence.split())
#for doc in nlp.pipe(sentences):

        
#        tokens.append([token.text for token in doc])
print('Done.')
        
with open(str(Path.cwd() / Path('data')) + '/star_trek_corpus.txt', 'w', encoding='utf-8') as corpus_file:
    for sentence in sentences:
        corpus_file.write(sentence + '\n')






