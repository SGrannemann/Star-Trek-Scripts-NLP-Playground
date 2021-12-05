"""Cleans the scripts of the episodes, adds plot summaries from Wikipedia and combines tokens into bigrams.
Adds the Wikipedia plot data (see data folder) to the dataframe.
Creates a serialized version of a dataframe that contains
both the cleaned text as well as the cleand text with bigrams."""
import sys
from pathlib import Path
import re
import pandas as pd
from pandas.core.frame import DataFrame
import spacy


def remove_speakers_and_empty_lines(episode_content: str) -> str:
    """Removes superfluous empty lines, the header of the scripts and the names of the speakers from the input data:
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
    # remove the standard header of the scripts. Those end with the air date of the episode
    # which itself ends with a year between 1986 (beginning of TNG) and 2001 (end of Voyager)
    complete_text_without_header = re.sub(r'^.*(19\d\d|20\d\d)\s{1,}', '', complete_text)
    return complete_text_without_header


def get_title(episode_content: str) -> str:
    """Function that extracts the title of an episode from the scripts.
    This allows finding the correct plot description from Wikipedia later on.
    Contains a lot of hardcoded cases because spelling etc. was very heterogenous."""
    # we need those boolean flags to know whether we already processed that episodes once
    # even though the use of the global is statement is suboptimal, this is a very easy
    # case --> I decided to use it here.
    global bestOfBoth  # pylint: disable=global-statement, invalid-name

    global chainOfCommand  # pylint: disable=global-statement, invalid-name

    title = ''
    for line in episode_content.split('\n'):
        # find the title
        if 'Transcripts' in line:
            title = line.split('-')[1].lower().strip()
    if 'part' in title:
        # this replaces all ' part ' partial strings with ', part'
        # which is the format used by the Wikipedia articles
        title = re.sub(r'\spart\s', ', part ', title)
        title = re.sub(r'\s1', ' i', title)
        title = re.sub(r'\s2', ' ii', title)
        # remove double comma - the transcripts are really inconsistent
        # in whether or not its Title, Part 1 or Title Part 1
        title = re.sub(r',,', ',', title)
    if 'honour' in title:
        # switch to american english spelling
        title = title.replace('honour', 'honor')
    if 'favour' in title:
        title = title.replace('favour', 'favor')
    if title == '' and series == 'tng':
        # somehow the script for the episode Peak Performance does not contain a title
        title = 'peak performance'
    if title == '' and series == 'voy':
        # somehow the script for the episode Scientific Method does not contain a title
        title = 'scientific method'

    # the following if statements are used to adjust the titles
    # from the scripts to the spelling of the Wiki articles.
    if title == 'all good things':
        title += '...'
    if title == 'who watches the':
        title += ' watchers'

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
    if title == 'the best of both' and bestOfBoth:  # pylint: disable=used-before-assignment
        title = 'the best of both worlds, part ii'
    if title == 'the best of both' and not bestOfBoth:
        title = 'the best of both worlds, part i'
        bestOfBoth = True
    if title == 'chain of command, part' and chainOfCommand:  # pylint: disable=used-before-assignment
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
    if title == 'let he who is without':
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


def create_dataframe_for_show(all_scripts: DataFrame, show: str) -> DataFrame:
    """Creates a DataFrame for the different shows, including removing speakers and
    empty lines, correcting the episode numbering and
    adding the plot descriptions from Wikipedia."""

    # remove the names of the speakers and get rid of the empty lines
    # and get the titles of the episodes
    if show == 'tng':
        show_scripts = all_scripts.TNG

    elif show == 'voy':
        show_scripts = all_scripts.VOY
        # TNG has more episodes than the other shows so the pd.Series objects for
        # the shorter shows have NaN values. Drop them.
        show_scripts = show_scripts.dropna()
    elif show == 'ds9':
        show_scripts = all_scripts.DS9
        show_scripts = show_scripts.dropna()
    else:
        print('Unknown show...')
        sys.exit()

    # get the title and text of the script
    show_scripts_cleaned = show_scripts.map(remove_speakers_and_empty_lines)
    show_titles = show_scripts.map(get_title)

    # combine the pd.Series to a dataframe.
    show_scripts_cleaned = pd.DataFrame({'EpisodeText': show_scripts_cleaned,
                                        'Title': show_titles}, index=show_scripts_cleaned.index)
    show_scripts_cleaned['Show'] = show.upper()
    # now get the wiki plot summaries
    return show_scripts_cleaned


if __name__ == '__main__':
    # paths for data
    PATH_TO_DATA = Path.cwd() / Path('data')

    # setup spacy language model
    nlp = spacy.load('en_core_web_sm')

    # read in scripts from the json file
    all_series_scripts = pd.read_json(PATH_TO_DATA / Path('all_scripts_raw.json'))

    # the following two booleans are used to know if the first part
    # of the double episodes has already been processed
    # those are two part episodes that were hard to handle.
    bestOfBoth = False  # pylint: disable=invalid-name
    chainOfCommand = False  # pylint: disable=invalid-name

    # create empty dataframe
    complete_frame = pd.DataFrame(columns=['EpisodeText', 'Title', 'Wiki_plot', 'Show'])
    # combine the empty frame with the frames from the different shows.
    # we build one DataFrame where each episode is a row
    # (so the first 176 rows should be TNG, followed by DS9 ,...)
    for series in ['tng', 'ds9', 'voy']:
        complete_frame = pd.concat([complete_frame, create_dataframe_for_show(
            all_series_scripts, series)], ignore_index=True)
    print(complete_frame.head())

    complete_frame.to_pickle(str(PATH_TO_DATA) / Path('scripts_and_plots.pkl'))
