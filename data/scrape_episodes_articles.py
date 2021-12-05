"""Webscrape the plot descriptions for the episodes from Memory Alpha Wiki."""
from pathlib import Path
import re
import requests
import bs4

BASE_URL = 'https://memory-alpha.fandom.com/'

# edit here to scrape other shows

# tng
#path_to_save_files = Path.cwd() / Path('data') / Path('scraped') / Path('tng')
#SERIES_TITLE_PATTERN = r' \(Star Trek: The Next Generation\)'
#response_object = requests.get('https://memory-alpha.fandom.com/wiki/Star_Trek:_The_Next_Generation')

# voy
path_to_save_files = Path.cwd() / Path('data') / Path('scraped') / Path('voy')
SERIES_TITLE_PATTERN = r' \(Star Trek: Voyager\)'
response_object = requests.get('https://memory-alpha.fandom.com/wiki/Star_Trek:_Voyager')

# ds9
#path_to_save_files = Path.cwd() / Path('data') / Path('scraped') / Path('ds9')
#SERIES_TITLE_PATTERN = r' \(Star Trek: Deep Space Nine\)'

#response_object = requests.get('https://memory-alpha.fandom.com/wiki/Star_Trek:_Deep_Space_Nine')


response_object.raise_for_status()
soup = bs4.BeautifulSoup(response_object.text, 'html.parser')


# find all links from the tables on the page and create URLs to open
# the links are in td elements with CSS class = summary
# .summary selects all class = summary elements,
# .summary a selects all a elements under .summary class elements
extracted_links = soup.select('.grey tr td:nth-of-type(1) > a')
links_to_episodes = [BASE_URL + extracted_link.get('href') for extracted_link in extracted_links]

episode_titles = [extracted_link.text for extracted_link in extracted_links]
# remove the series title patterns. these are only necessary for disambiguation on the Wiki itself
# (i.e. there might be entities with the same name as the title of Star Trek episode)
episode_titles_no_series_title = [re.sub(SERIES_TITLE_PATTERN, '', title) for title in episode_titles]
# remove : and ? from titles to prevent messing up file saving.
episode_titles_no_series_title = [re.sub(r':|\?', ' ', title) for title in episode_titles_no_series_title]
titles_and_links = list(zip(episode_titles_no_series_title, links_to_episodes))

if not path_to_save_files.exists():
    path_to_save_files.mkdir(parents=True)
# for each of these links save the webpage it points to
for (title, link) in titles_and_links:
    res_ob = requests.get(link)
    res_ob.raise_for_status()
    with open(str(path_to_save_files / Path(title)) + '.txt', 'w', encoding='utf-8') as episode_file:
        episode_file.write(res_ob.text)
