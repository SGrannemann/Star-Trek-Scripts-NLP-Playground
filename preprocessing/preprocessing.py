"""Cleans the scripts of the episodes and combines tokens into bigrams. Adds the Wikipedia plot data (see data folder) to the dataframe.
Creates a serialized version of a dataframe that contains both the cleaned text as well as the cleand text with bigrams."""
import pandas as pd
from gensim.models.phrases import Phraser
from nltk.tokenize import sent_tokenize
from pathlib import Path
import re
from pandas.core.frame import DataFrame
import spacy
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: Add Column for the show to the dataframe.
# TODO: Improve function DocStrings.
# TODO: Evaluate for DS9.
# TODO: Evalute for VOY



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
    """Function that extracts the title of an episode from the scripts. 
    This allows finding the correct plot description from Wikipedia later on.
    Contains a lot of hardcoded cases because spelling etc. was very heterogenous."""
    # we need those boolean flags to know whether we already processed that episodes once
    global bestOfBoth
    global chainOfCommand

    title = ''
    for line in episode_content.split('\n'):
        # find the title
        if 'Transcripts' in line:
            title = line.split('-')[1].lower().strip()
    if 'part' in title:
        # this replaces all ' part ' partial strings with ', part' which is the format used by the Wikipedia articles
        title = re.sub(r'\spart\s', ', part ', title)
        title = re.sub(r'\s1', ' i', title)
        title = re.sub(r'\s2', ' ii', title)
    if 'honour' in title:
        # switch to american english spelling
        title = title.replace('honour', 'honor')

    if title == '':
        # somehow the script for the episode Peak Performance does not contain a title
        title = 'peak performance'
    
    # the following if statements are used to adjust the titles from the scripts to the spelling of the Wiki articles.
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


def combine_to_bigrams(complete_text: str) -> str:
    """function that combines bigrams from the text where appropriate. Returns string with combined bigrams."""
    sentences = sent_tokenize(complete_text)
    bigram_episode = []
    for sentence in sentences:
        tokens = [token.lemma_ for token in nlp(sentence) if token.text not in [' ', '\n']]
        bigram_tokens = bigrams[tokens]
        bigram_episode.append(' '.join(bigram_tokens))
    return ' '.join(bigram_episode)


def get_wiki_plots(show_scripts: DataFrame, show:str) -> DataFrame:
    """Adds the Wiki plot descriptions to the dataframe and returns it with plot descriptions."""
    # set the index of the dataframe to title for easy combination with the Wiki articles later on
    show_scripts.set_index('Title', inplace=True)

    # construct path to saved data
    folder_for_saving = Path.cwd() / Path('data') / Path('scraped') / Path(show) / Path('processed')

    # variables to keep track of the wiki articles we read in
    plots = []
    titles = []
    # grab the plot descriptions from the files
    for plot_description_file in folder_for_saving.glob('*.txt'):
        title = plot_description_file.stem
        with open(plot_description_file, 'r', encoding='utf-8') as episode_file:
            plot = episode_file.read()
            titles.append(title.lower())
            plots.append(plot)

    # add the plot descriptions to the dataframe        
    wiki_plots = pd.Series(plots, index=titles, dtype='string')
    show_scripts = show_scripts.assign(Wiki_plot=wiki_plots)

    # check for bad values
    if show_scripts.Wiki_plot.isnull().sum() != 0:
        print('NaN entries found. Check your data!')
        print('Wiki plots without match for {}:'.format(show))
        print(show_scripts[show_scripts.Wiki_plot.isnull()])
    
    # reset the index
    show_scripts.reset_index(inplace=True)

    return show_scripts

def create_dataframe_for_show(all_scripts: DataFrame, show:str) -> DataFrame:
    """Creates a DataFrame for the different shows, including removing speakers and empty lines, correcting the episode numbering and
    adding the plot descriptions from Wikipedia."""

    # remove the names of the speakers and get rid of the empty lines and get the titles of the episodes
    if show == 'tng':
        
        show_scripts_cleaned = all_scripts.TNG.map(remove_speakers_and_empty_lines)
        show_titles = all_scripts.TNG.map(get_title)
        # the index does not match the episode number: both the first and last episode have one script (thus only one row in the dataframe), but count as two episodes each. 
        # i.e. the first episode after the initial two-part episode has number 3
        # to be able to easily access the correct episode_number later on, we add a new column with the correct episode number
        
    elif show == 'voy':
        show_scripts_cleaned = all_scripts.VOY.map(remove_speakers_and_empty_lines)
        show_titles = all_scripts.VOY.map(get_title)
    elif show == 'ds9':
        show_scripts_cleaned = all_scripts.DS9.map(remove_speakers_and_empty_lines)
        show_titles = all_scripts.DS9.map(get_title)
    else:
        print('Unknown show...')
        exit()
    true_episode_numbers = [i for i in range(1,178)]
    true_episode_numbers.remove(2)
    true_episode_numbers_series = pd.Series(true_episode_numbers, index=show_scripts_cleaned.index)
    show_scripts_cleaned = pd.DataFrame({'EpisodeText' : show_scripts_cleaned, 'Title' : show_titles, 'Episode_Number': true_episode_numbers_series}, index=show_scripts_cleaned.index)
    show_scripts_cleaned_wiki = get_wiki_plots(show_scripts_cleaned, show)
    return show_scripts_cleaned_wiki

if __name__ == '__main__':
    # paths for data
    PATH_TO_DATA = Path.cwd() / Path('data')
    
    # setup spacy language model
    nlp = spacy.load('en_core_web_sm')

    # read in scripts from the json file
    all_series_scripts = pd.read_json(PATH_TO_DATA / Path('all_scripts_raw.json'))
    all_series_scripts.fillna('', inplace=True)
    
    # the following two booleans are used to know if the first part of the double episodes has already been processed
    # those are two part episodes that were hard to handle.
    bestOfBoth = False
    chainOfCommand = False

    # create empty dataframe
    complete_frame = pd.DataFrame(columns=['EpisodeText', 'Title', 'Episode_Number', 'Wiki_plot'])
    # combine the empty frame with the frames from the different shows.
    for show in ['tng', 'voy', 'ds9']:
        complete_frame = pd.concat([complete_frame, create_dataframe_for_show(all_series_scripts, show)], ignore_index=True)
    print(complete_frame.head())
    
    
    


    # use the bigram model to combine tokens into bigrams if appropriate
    # bigrams = Phraser.load(str(PATH_TO_DATA) + '\\bigram_model.pkl')
    
    # # combine the script and the plot description
    # tng_series_scripts_cleaned['complete_text'] = tng_series_scripts_cleaned['EpisodeText'] + tng_series_scripts_cleaned['wiki_plot']
    # # lemmatize them and remove empty strings as well as newline chars.
    
    # tng_series_scripts_cleaned['complete_text_with_bigrams'] =  tng_series_scripts_cleaned['complete_text'].apply(combine_to_bigrams)
    
    


    #tng_series_scripts_cleaned.to_pickle(str(PATH_TO_DATA) + '\cleaned_tng_scripts.pkl')
