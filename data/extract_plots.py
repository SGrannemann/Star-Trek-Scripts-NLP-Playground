"""Module for extracting the plot descriptions from the
scraped Memory Alpha articles and writing them to a simple text file."""
from pathlib import Path
import re
import bs4
from bs4.element import NavigableString


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
        # lets find the header for plot, but we only want to get the Act wise descriptions, not the teaser
        act_headings = soup.find_all(id=re.compile('^Act'))

        for act in act_headings:  # iterate over all acts
            # this line allows us to iterate over all elements that come next/under:
            for para_element in act.parent.nextSiblingGenerator():
                # ignore h3, h2 because this is either the next act or we are done with the plot description
                if para_element.name in ['h3', 'h2']:
                    break
                if para_element.name == 'figure':  # we dont want the captions
                    continue
                if isinstance(para_element, NavigableString):
                    continue
                if para_element.text:

                    plot.append(para_element.text)

        # process a bit and save to new file.
        if not FOLDER_FOR_SAVING.exists():
            FOLDER_FOR_SAVING.mkdir(parents=True)
        with open(FOLDER_FOR_SAVING / Path(title), 'w', encoding='utf-8') as file_to_write:
            # lets remove the parts that are in brackets - these are usually the names of the actors
            COMPLETE_TEXT = ' '.join(plot)
            # Regex: Take everything between ( and ) with non greedy matching
            text_without_actors = re.sub(r'\(.{1,}?\)', '', COMPLETE_TEXT)
            file_to_write.write(text_without_actors)
