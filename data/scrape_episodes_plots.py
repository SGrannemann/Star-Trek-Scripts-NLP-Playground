"""Webscrape the plot descriptions for the episodes from Wikipedia."""

import requests
import bs4

from pathlib import Path
import re

path_to_save_files = Path.cwd() / Path('data') / Path('scraped') / Path('tng')
SERIES_TITLE_PATTERN = r' \(Star Trek: The Next Generation\)' 

# open the list of next gen episodes and convert to bs4 
response_object = requests.get('https://en.wikipedia.org/wiki/List_of_Star_Trek:_The_Next_Generation_episodes')
response_object.raise_for_status()
soup = bs4.BeautifulSoup(response_object.text, 'html.parser')


# find all links from the tables on the page and create URLs to open
# the links are in td elements with CSS class = summary
# .summary selects all class = summary elements, 
# .summary a selects all a elements under .summary class elements
extracted_links = soup.select('.summary a')
links_to_episodes = ['http://en.wikipedia.org' + extracted_link.get('href') for extracted_link in extracted_links]

episode_titles = [extracted_link.text for extracted_link in extracted_links]
episode_titles_no_series_title = [re.sub(SERIES_TITLE_PATTERN, '', title) for title in episode_titles]
titles_and_links = list(zip(episode_titles_no_series_title, links_to_episodes))

# for each of these links save the webpage it points to
for (title, link) in titles_and_links:
    res_ob = requests.get(link)
    res_ob.raise_for_status()
    with open(str(path_to_save_files /  Path(title)) + '.txt', 'w', encoding='utf-8') as episode_file:
        episode_file.write(res_ob.text)







# save the pages in files (name should be the episodes title)

# extract the plot descriptions into their own files

