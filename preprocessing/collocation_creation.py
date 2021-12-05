""" Script to create ngrams based on their collocations and a
file for the complete corpus in a format suitable to train gensim word2vec on it."""
from pathlib import Path
import pandas as pd
import spacy
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from nltk.tokenize import sent_tokenize

# read in the scripts and wikipedia plot descriptions
scripts_and_plots = pd.read_pickle(str(Path.cwd() / Path('data')) + '/scripts_and_plots.pkl')


# first read in the text from Memory Alpha and add it all together.
path_to_data = Path.cwd() / Path('data') / Path('scraped')
episode_folders = ['ds9', 'tng', 'voy']
text = ''  # pylint: disable=invalid-name
for series in episode_folders:
    final_path = path_to_data / Path(series) / Path('processed')
    for episode_file in final_path.glob('*.txt'):
        with open(episode_file, 'r', encoding='utf-8') as file:
            text += file.read()

# read in the scripts
for index, episode_text in scripts_and_plots.EpisodeText.items():
    text += episode_text


nlp = spacy.load('en_core_web_sm')
# prepare for collocation determination by tokenising and lemmatization
sentences = sent_tokenize(text)
wordtokenized_lemmatized_sentences = []
for sent in nlp.pipe(sentences, disable=['ner', 'parser']):

    wordtokenized_lemmatized_sentences.append([token.lemma_ for token in sent if token.text != ' '])

# train the actual bigram model

bigrams = Phrases(wordtokenized_lemmatized_sentences)


# lets save that model to use in later steps

bigram_model = Phraser(bigrams)
bigram_model.save(str(path_to_data / Path('bigram_model.pkl')))

# create a corpus file with bigrams for later training of word2vec.

with open('star_trek_corpus.txt', 'w', encoding='utf-8') as corpus_file:
    for sentence in wordtokenized_lemmatized_sentences:
        corpus_file.write(' '.join(bigrams[sentence]) + '\n')
