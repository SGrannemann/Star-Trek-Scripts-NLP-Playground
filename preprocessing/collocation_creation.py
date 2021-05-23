""" Script to create ngrams based on their collocations."""
import pandas as pd
from pathlib import Path
from gensim.models.phrases import Phrases
from gensim.models.phrases import Phraser
from nltk.tokenize import sent_tokenize, word_tokenize

# first read in the text and add it all together.
path_to_data = Path.cwd() / Path('data')
tng_series_scripts_cleaned = pd.read_pickle(path_to_data / Path('cleaned_tng_scripts.pkl'))
text = ''
for index, episode_text in tng_series_scripts_cleaned.EpisodeText.items():
    text += episode_text

# prepare for collocation determination
sentences = sent_tokenize(text)
word_tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

bigrams = Phrases(word_tokenized_sentences)




# lets save that model to use in later steps

bigram_model = Phraser(bigrams)
bigram_model.save(str(path_to_data / Path('bigram_model.pkl')))