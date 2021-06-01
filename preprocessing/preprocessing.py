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
            cleaned_lines.append(part_to_keep.strip())
            continue
        # after this string there are only information about the franchise, we can leave those out.
        if line == '<Back':
            break
        cleaned_lines.append(line.strip())

    complete_text = ' '.join(cleaned_lines)
    # remove the standard header of the scripts
    complete_text_without_header = re.sub(r'^.*(19\d\d|20\d\d)\s{1,}', '', complete_text)
    return complete_text_without_header

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
        # remove double comma - the transcripts are really inconsistent in whether or not its Title, Part 1 or Title Part 1 
        title = re.sub(r',,', ',', title)
    if 'honour' in title:
        # switch to american english spelling
        title = title.replace('honour', 'honor')
    if 'favour' in title:
        title = title.replace('favour', 'favor')
    if title == '' and show == 'tng':
        # somehow the script for the episode Peak Performance does not contain a title
        title = 'peak performance'
    if title == '' and show == 'voy':
        # somehow the script for the episode Scientific Method does not contain a title
        title = 'scientific method'

    # the following if statements are used to adjust the titles from the scripts to the spelling of the Wiki articles.
    if title == 'all good things':
        title += '...'
    if title == 'who watches the':
        title += ' watchers'
    #if title == 'q who?':
    #    title = title.strip('?')
    if '?' in title:
        title = re.sub(r'\?', '', title)
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
    if title == 'course: oblivion':
        title = 'course oblivion'
    if title == '11:59':
        title = '11 59'
    if title == 'vis a vis':
        title = 'vis à vis'
    if title == 'q':
        title = 'q-less'
    if title == 'in the hands of the':
        title = 'in the hands of the prophets'
    if title == 'through the looking':
        title = 'through the looking glass'
    if title == 'the sons of mogh':
        title = 'sons of mogh'
    if title == "looking for par'mach":
        title = "looking for par'mach in all the wrong places"
    if title == 'nor the battle to the':
        title = '...nor the battle to the strong'
    if title == 'trials and':
        title = 'trials and tribble-ations'
    if title ==  'let he who is without':
        title += ' sin...'
    if title == 'the darkness and the':
        title += ' light'
    if title == 'doctor bashir, i':
        title += ' presume'
    if title == 'you are cordially':
        title += ' invited'
    if title == 'statistical':
        title += ' probabilities'
    if title == 'wrongs darker than':
        title += ' death or night'
    if title == 'take me out to the':
        title += ' holosuite'
    if title == 'treachery, faith and':
        title += ' the great river'
    if title == 'once more unto the':
        title += ' breach'
    if title == 'siege of ar':
        title = 'the siege of ar-558'
    if title == 'badda':
        title = 'badda-bing badda-bang'
    if title == 'inter arma enim silent':
        title += ' leges'
    if title == 'till death us do part':
        title = "'til death do us part"
    if title == 'the changing face of':
        title += ' evil'
    if title == 'improbable cause':
        title = 'improbable cause'
    return title


def combine_to_bigrams(complete_text: str) -> str:
    """function that combines bigrams from the text where appropriate. Returns string with combined bigrams."""
    sentences = sent_tokenize(complete_text)
    bigram_episode = []
    for sentence in nlp.pipe(sentences, disable=['ner', 'parser']):
        tokens = [token.lemma_ for token in sentence if token.text != ' ']
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
            # map the scripts from DS9 for two-part episodes to the correct wiki entry. Somehow in the Wiki entry these were done differently
            # than for the other shows. e.g. : For TNG the wiki entry always states "Title, Part I" / "Title, Part, II". For DS9 it is "Title," "Part I" :(
            if title.lower() == 'the maquis':
                titles.append('the maquis, part i')
                titles.append('the maquis, part ii')
                plots.append(plot)
                plots.append(plot)
            elif title.lower() == 'the search':
                titles.append('the search, part i')
                titles.append('the search, part ii')
                plots.append(plot)
                plots.append(plot)
            elif title.lower() == 'past tense':
                titles.append('past tense, part i')
                titles.append('past tense, part ii')
                plots.append(plot)
                plots.append(plot)
            else:
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
        show_scripts = all_scripts.TNG
      
        
    elif show == 'voy':

        show_scripts = all_scripts.VOY
        # TNG has more episodes than the other shows so the pd.Series objects for the shorter shows have NaN values. Drop them.
        show_scripts = show_scripts.dropna()
        

    elif show == 'ds9':
        show_scripts = all_scripts.DS9
        show_scripts = show_scripts.dropna()
       
        

    else:
        print('Unknown show...')
        exit()

    show_scripts_cleaned = show_scripts.map(remove_speakers_and_empty_lines)
    show_titles = show_scripts.map(get_title)
    
    

    show_scripts_cleaned = pd.DataFrame({'EpisodeText' : show_scripts_cleaned, 'Title' : show_titles}, index=show_scripts_cleaned.index)
    show_scripts_cleaned['Show'] = show.upper()
    show_scripts_cleaned_wiki = get_wiki_plots(show_scripts_cleaned, show)
    return show_scripts_cleaned_wiki

if __name__ == '__main__':
    # paths for data
    PATH_TO_DATA = Path.cwd() / Path('data')
    
    # setup spacy language model
    nlp = spacy.load('en_core_web_sm')

    # read in scripts from the json file
    all_series_scripts = pd.read_json(PATH_TO_DATA / Path('all_scripts_raw.json'))
    
    
    # the following two booleans are used to know if the first part of the double episodes has already been processed
    # those are two part episodes that were hard to handle.
    bestOfBoth = False
    chainOfCommand = False

    # create empty dataframe
    complete_frame = pd.DataFrame(columns=['EpisodeText', 'Title', 'Wiki_plot', 'Show'])
    # combine the empty frame with the frames from the different shows. 
    # we build one DataFrame where each episode is a row (so the first 176 rows should be TNG, followed by DS9 ,...)
    for show in ['tng', 'ds9', 'voy']:
        complete_frame = pd.concat([complete_frame, create_dataframe_for_show(all_series_scripts, show)], ignore_index=True)
    print(complete_frame.head())
    
    
    


    # use the bigram model to combine tokens into bigrams if appropriate
    bigrams = Phraser.load(str(PATH_TO_DATA) + '\\bigram_model.pkl')
    
    #  combine the script and the plot description
    print('Combining texts...')
    complete_frame['complete_text'] = complete_frame['EpisodeText'] + complete_frame['Wiki_plot']
    # # lemmatize them and remove empty strings as well as newline chars.
    
    print('Combining bigrams...')
    complete_frame['complete_text_with_bigrams'] =  complete_frame['complete_text'].apply(combine_to_bigrams)
    
    


    complete_frame.to_pickle(str(PATH_TO_DATA) + '\scripts_and_plots.pkl')
