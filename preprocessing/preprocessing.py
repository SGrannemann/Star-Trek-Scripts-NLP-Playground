"""Cleans the scripts of the episodes and combines tokens into bigrams.
Creates a serialized version of a dataframe that contains both the cleaned text as well as the cleand text with bigrams."""
import pandas as pd
from gensim.models.phrases import Phraser
from nltk.tokenize import word_tokenize
from pathlib import Path
import re
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: Add plot descriptions from wikipedia to dataframe.
PATH_TO_DATA = Path.cwd() / Path('data')
FOLDER_FOR_SAVING = Path.cwd() / Path('data') / Path('scraped') / Path('tng') / Path('processed')
# the following two booleans are used to know if the first part of the double episodes has already been processed
bestOfBoth = False
chainOfCommand = False
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
    global bestOfBoth
    global chainOfCommand
    title = ''
    for line in episode_content.split('\n'):
        # find the title
        if 'Transcripts' in line:
            title = line.split('-')[1].lower().strip()
    if 'part' in title:
        # TODO: Finish RegEx to add a comma before part
        title = re.sub(r'\spart\s', ', part ', title)
        title = re.sub(r'\s1', ' i', title)
        title = re.sub(r'\s2', ' ii', title)
    if 'honour' in title:
        title = title.replace('honour', 'honor')

    if title == '':
        title = 'peak performance'
    
    if title == 'all good things':
        title += '...'
    if title == 'who watches the':
        title += ' watchers'
    if title == 'q who?':
        title = title.strip('?')
    if title == 'menage a troi':
        title = 'ménage à troi'
    if title == 'redemption':
        title = 'redemption, part i'
    if title == 'time\'s arrow,, part i':
        title = 'time\'s arrow, part i'
    if title == 'time\'s arrow, part two':
        title = 'time\'s arrow, part ii'
    if title == 'true':
        title = 'true q'
    if title == 'the best of both' and bestOfBoth:
        title = 'the best of both worlds, part ii'
    if title == 'the best of both' and not bestOfBoth:
        title = 'the best of both worlds, part i'
        bestOfBoth = True
    if title == 'chain of command, part' and chainOfCommand:
        title = 'chain of command, part ii'
    if title == 'chain of command, part' and not chainOfCommand:
        title = 'chain of command, part i'
        chainOfCommand = True
    return title

all_series_scripts = pd.read_json(PATH_TO_DATA / Path('all_scripts_raw.json'))

# remove the names of the speakers and get rid of the empty lines

tng_series_scripts_cleaned = all_series_scripts.TNG.map(remove_speakers_and_empty_lines)
tng_series_scripts_cleaned = pd.DataFrame({'EpisodeText' : tng_series_scripts_cleaned})
tng_series_scripts_cleaned['title'] = all_series_scripts.TNG.map(get_title)

print(tng_series_scripts_cleaned.head(10))
tng_series_scripts_cleaned.set_index('title', inplace=True)


plots = []
titles = []

for plot_description_file in FOLDER_FOR_SAVING.glob('*.txt'):
    title = plot_description_file.stem
    with open(plot_description_file, 'r') as episode_file:
        plot = episode_file.read()
        titles.append(title.lower())
        plots.append(plot)
wiki_plots = pd.Series(plots, index=titles, dtype='string')
tng_series_scripts_cleaned = tng_series_scripts_cleaned.assign(wiki_plot=wiki_plots)
print(tng_series_scripts_cleaned.head(100))
nan_frame = tng_series_scripts_cleaned[tng_series_scripts_cleaned.isna().any(axis=1)]
print(nan_frame)
print(tng_series_scripts_cleaned.wiki_plot.isnull().sum())

# use the bigram model to combine tokens into bigrams if appropriate
#bigrams = Phraser.load(str(PATH_TO_DATA) + '\\bigram_model.pkl')
# TODO: text with bigrams should first concat script and plot description, then use the phraser.
#stng_series_scripts_cleaned['text with bigrams'] =  [' '.join(bigrams[word_tokenize(episode_text)]) for episode_text in tng_series_scripts_cleaned.EpisodeText]
# get the title of the episode in an extra column


tng_series_scripts_cleaned.to_pickle('cleaned_tng_scripts.pkl')
