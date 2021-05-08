"""Cleans the scripts of the episodes and combines tokens into bigrams.
Creates a serialized version of a dataframe that contains both the cleaned text as well as the cleand text with bigrams."""
import pandas as pd
from gensim.models.phrases import Phraser
from nltk.tokenize import word_tokenize
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')



def remove_speakers_and_empty_lines(episode_content: str) -> str:
    """Removes superfluous empty lines and the names of the speakers from the input data:
    e.g. :
    Picard: Make it so.
    becomes:
    Make it so.
    """

    cleaned_lines = []
  

    for line in episode_content.split('\n'):
    
        # ignore empty lines
        if line == '':
            continue
        # lines that start with square brackets are just information about the location.
        if line.startswith('['):
            continue
        # the actual talking lines always contain a ':' - we will just keep the text, not the talker
        # for this application
        if ':' in line:
            part_to_keep = line.split(':')[1]
            cleaned_lines.append(part_to_keep.strip() + ' \n')
            continue
        # after this string there are only information about the franchise, we can leave those out.
        if line == '<Back':
            break
        cleaned_lines.append(line.strip() + ' \n')
    return ''.join(cleaned_lines)

def get_title(episode_content:str) -> str:
    title = ''
    for line in episode_content.split('\n'):
        # find the title
        if 'Transcripts' in line:
            title = line.split('-')[1]
    if title == '':
        title = 'Peak Performance'
    return title

all_series_scripts = pd.read_json('all_scripts_raw.json')
#print(all_series_scripts.TNG)
#  remove the names of the speakers and get rid of the empty lines
# and I'll focus on The Next Generation Episodes for now
tng_series_scripts_cleaned = all_series_scripts.TNG.map(remove_speakers_and_empty_lines)
tng_series_scripts_cleaned = pd.DataFrame({'EpisodeText' : tng_series_scripts_cleaned})

# use the bigram model to combine tokens into bigrams if appropriate
bigrams = Phraser.load('bigram_model.pkl')

tng_series_scripts_cleaned['text with bigrams'] =  [' '.join(bigrams[word_tokenize(episode_text)]) for episode_text in tng_series_scripts_cleaned.EpisodeText]
# get the title of the episode in an extra column
tng_series_scripts_cleaned['title'] = all_series_scripts.TNG.map(get_title)

tng_series_scripts_cleaned.to_pickle('cleaned_tng_scripts.pkl')
