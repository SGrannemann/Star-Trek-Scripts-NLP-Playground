# Star-Trek-Scripts-NLP-Playground
## Project as a training ground for different aspects of NLP
This project is intended as a toy project to practice different aspects of NLP, from collecting and preprocessing data to the options for search and question answering. I was especially interested in trying out the Haystack library.
You can use the provided Python scripts to create your own corpus of Memory Alpha plot summaries and afterwards query the corpus. For now, I have focussed more on collecting and cleaning the data instead of using advanced NLP methods. Available query options are thus limited to the Haystack implementation.

## Getting Started
In case you want to use the code available here, you can most easily proceed as follows:
- Use the requirements.txt to setup your VENV with the necessary libraries.
- Download the Memory Alpha articles via the provided scraping script (data folder: scrape_episodes_articles.py). 
- Use the script extract_plots.py from the data folder to write the plot descriptions from the Wiki articles to text files.
- Train the bigram model and save a corpus for word2vec via collocation_creation.py.
- Run word2vec_creation.py to create the word2vec vector embedding model of the corpus and try it out if you want ;).
- You can now use the combined_search_qa.py script to query the corpus based on TFIDF or DPR methods and ask questions. Have fun - and do not hesitate to give feedback :)


## Project structure

This project comes with multiple folders to organize all the files. These folders are:
- Data
- Preprocessing

### Data
In this folder, you can find the raw data as well as artifacts generated for the different steps of the projects: Serialized DataFrames, collocation models etc.
Scripts for collecting additional data can be found here as well, such as scraping episode articles from Wikipedia.


### Preprocessing
In this folder, you can find scripts that preprocess the data, either for extracting the text from the original files or for cleaning up text data etc. 
Additionally, modules that create collocations or word2vec embeddings reside in this folder.

## Contact and contribution
Should you want to contribute or to contact me, feel free to send me an email.
Live long and prosper!





## Data collection and cleaning
This playground currently uses the following data:
- A dataset with the scripts of the episodes, taken from here: https://www.kaggle.com/gjbroughton/start-trek-scripts
For this part of the data, functions that can extract the episode title from the scripts as well as clean up (remove speakers for example) the text are available.

- A dataset of plot descriptions that I scraped from the Wikipedia articles. The scripts for downloading the files and extracting the actual descriptions of the plots can be found in the data folder.
This part of the data was scraped from Wikipedia. The Plot Descriptions of the episodes were extracted/parsed with a simple script.


## 



