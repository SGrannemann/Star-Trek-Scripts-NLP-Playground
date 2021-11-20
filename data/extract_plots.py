"""Module for extracting the plot descriptions from the
scraped Wikipedia articles and writes them to a simple text file."""
from pathlib import Path
import re
import bs4



# set the series folder here

# tng folders
FOLDER_TO_WIKIDATA = Path.cwd() / Path('data') / Path('scraped') / Path('tng')
FOLDER_FOR_SAVING = Path.cwd() / Path('data') / Path('scraped') / Path('tng') / Path('processed')

# voy folders
#FOLDER_TO_WIKIDATA = Path.cwd() / Path('data') / Path('scraped') / Path('voy')
#FOLDER_FOR_SAVING = Path.cwd() / Path('data') / Path('scraped') / Path('voy') / Path('processed')

# ds9 folders
#FOLDER_TO_WIKIDATA = Path.cwd() / Path('data') / Path('scraped') / Path('ds9')
#FOLDER_FOR_SAVING = Path.cwd() / Path('data') / Path('scraped') / Path('ds9') / Path('processed')


# loop over all text files, find the plot element and get all para elements
# under that until we hit the next heading
for wikipage in FOLDER_TO_WIKIDATA.glob('*.txt'):
    with open(wikipage, 'r', encoding='utf-8') as episode_document:
        title = wikipage.name

        soup = bs4.BeautifulSoup(episode_document, 'html.parser')
        plot = []
        # lets find the header for plot, but we only want to get the Act wise descriptions, not the teaser
        act_headings = soup.find_all(id=re.compile('^Act'))
        

        for act in act_headings: # iterate over all acts
            for para_element in act.parent.nextSiblingGenerator(): # this line allows us to iterate over all elements that come next/under
                # to an act heading
                if para_element.name in ['h3', 'h2']: # this is either the next act or we are done with the plot description
                    break
                if para_element.name == 'figure': # we dont want the captions
                    continue
                if para_element == '\n' or ' ':
                    continue
                if para_element.text:
                    plot.append(para_element.text)

        # process a bit and save to new file.
        if not FOLDER_FOR_SAVING.exists():
            FOLDER_FOR_SAVING.mkdir(parents=True)
        with open(FOLDER_FOR_SAVING / Path(title), 'w', encoding='utf-8') as file_to_write:
            # lets remove the parts that are in brackets - these are usually the names of the actors
            complete_text = ' '.join(plot)
            # Regex: Take everything between ( and ) with non greedy matching
            text_without_actors = re.sub(r'\(.{1,}?\)', '', complete_text)
            file_to_write.write(text_without_actors)
            