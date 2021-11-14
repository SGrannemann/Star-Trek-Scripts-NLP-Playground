"""Module for extracting the plot descriptions from the
scraped Wikipedia articles and writes them to a simple text file."""
from pathlib import Path
import re
import bs4



# set the series folder here

# tng folders
#FOLDER_TO_WIKIDATA = Path.cwd() / Path('data') / Path('scraped') / Path('tng')
#FOLDER_FOR_SAVING = Path.cwd() / Path('data') / Path('scraped') / Path('tng') / Path('processed')

# voy folders
#FOLDER_TO_WIKIDATA = Path.cwd() / Path('data') / Path('scraped') / Path('voy')
#FOLDER_FOR_SAVING = Path.cwd() / Path('data') / Path('scraped') / Path('voy') / Path('processed')

# ds9 folders
FOLDER_TO_WIKIDATA = Path.cwd() / Path('data') / Path('scraped') / Path('ds9')
FOLDER_FOR_SAVING = Path.cwd() / Path('data') / Path('scraped') / Path('ds9') / Path('processed')


# loop over all text files, find the plot element and get all para elements
# under that until we hit the next heading
for wikipage in FOLDER_TO_WIKIDATA.glob('*.txt'):
    with open(wikipage, 'r', encoding='utf-8') as episode_document:
        title = wikipage.name

        soup = bs4.BeautifulSoup(episode_document, 'html.parser')
        plot = []
        # lets find the header for plot
        plot_start = soup.find(id='Plot')
        # Wikipedia is not always consistent in naming the sections
        if plot_start is None:
            plot_start = soup.find(id='Plot_summary')
        # iterate over all paragraph elements until we hit the next heading
        # this is the plot description
        for para_element in plot_start.parent.nextSiblingGenerator():
            if para_element.name == 'h2':
                break
            if para_element == '\n':
                continue
            if para_element.text:
                plot.append(para_element.text)

        # process a bit and save to new file.
        with open(FOLDER_FOR_SAVING / Path(title), 'w', encoding='utf-8') as file_to_write:
            # lets remove the parts that are in brackets - these are usually the names of the actors
            complete_text = ' '.join(plot)
            # Regex: Take everything between ( and ) with non greedy matching
            text_without_actors = re.sub(r'\(.{1,}?\)', '', complete_text)
            file_to_write.write(text_without_actors)
            