# Star-Trek-Scripts-NLP-Playground
## Project as a training ground for different aspects of NLP
This project is intended as a toy project to practice different aspects of NLP, from collecting and preprocessing data to the different vector space transformations and different options for scoring document similarity.
You can use the provided Python scripts to create your own corpus of Star Trek scripts and Wikipedia plot summaries and afterwards query the corpus. For now, I have focussed more on collecting and cleaning the data instead of using advanced NLP methods. Available query options are thus TFIDF, LSA, word2vec and doc2vec for now. All query options are very basic and no sophisticated algorithms.

## Getting Started
In case you want to use the code available here, you can most easily proceed as follows:
- Use the requirements.txt to setup your VENV with the necessary libraries.
- Download the all_scripts_raw.json file from the data folder of this project or directly from the source at https://www.kaggle.com/gjbroughton/start-trek-scripts .
- Download the Wikipedia articles via the provided scraping script (data folder: scrape_episode_plots.py). Please do not overburden Wikipedia while scraping - they are nice enough to be pretty lenient regarding webscraping.
- Use the script extract_plots.py from the data folder to write the plot descriptions from the Wiki articles to text files.
- now you can use the preprocessing.py script (in the preprocessing folder) that will create a pandas DataFrame with both the scripts and the plot descriptions. You will need to comment out the lines with the bigrams towards the end.
- Train the bigram model via collocation_creation.py.
- use preprocessing again so the DataFrame contains the bigram version of the text too. (Uncomment the respective lines again.)
- Run word2vec_doc2vec_creation.py to create the word2vec vector embedding model of the corpus and the doc2vec embeddding model.
- You can now use the query_scoring.py script to query the corpus based on TFIDF, LSA, word2vec and doc2vec. Have fun - and do not hesitate to give feedback :)


## Project structure

This project comes with multiple folders to organize all the files. These folders are:
- Analysis
- Data
- Preprocessing

### Analysis
In this folder, you can find an IPython Notebook for an initial, simple analysis of the data as well as results for optimizing hyperparameters, e.g. optimal number of dimensions for LSA.

### Data
In this folder, you can find the raw data as well as artifacts generated for the different steps of the projects: Serialized DataFrames, collocation models etc.
Scripts for collecting additional data can be found here as well, such as scraping episode articles from Wikipedia.


### Preprocessing
In this folder, you can find scripts that preprocess the data, either for extracting the text from the original files or for cleaning up text data etc. 
Additionally, modules that create collocations or TFIDF/LSA models or word2vec/doc2vec embeddings reside in this folder.

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



