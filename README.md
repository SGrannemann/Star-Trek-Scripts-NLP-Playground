# Star-Trek-Scripts-NLP-Playground
## Project as a training ground for different aspects of NLP
This project is intended as a toy project to practice different aspects of NLP, from collecting and preprocessing data to the different vector space transformations and different options for scoring document similarity.

## Getting Started
In case you want to use the code available here, you can most easily proceed as follows:
- Download the Wikipedia articles via the provided scraping script (data folder). Please do not overburden Wikipedia while scraping - they are nice enough to be pretty lenient regarding webscraping.
- Use the script extract_plots.py to write the plot descriptions from the Wiki articles to text files
- now you can use the preprocessing.py script (in the preprocessing folder) that will create a pandas DataFrame with both the scripts and the plot descriptions.
- Train the bigram model via collocation_creation.py.
- use preprocessing again so the DataFrame contains the bigram version of the text too.


## Project structure

This project comes with multiple folders to organize all the files. These folders are:
- Analysis
- Data
- Preprocessing

### Analysis
In this folder, you can find multiple IPython Notebooks for an initial, simple analysis of the data as well as results for optimizing hyperparameters, e.g. optimal number of dimensions for LSA.

### Data
In this folder, you can find the raw data as well as artifacts generated for the different steps of the projects: Serialized DataFrames, collocation models etc.
Scripts for collecting additional data can be found here as well, such as scraping episode articles from Wikipedia.


### Preprocessing
In this folder, you can find scripts that preprocess the data, either for extracting the text from the original files or for cleaning up text data etc. 
Additionally, modules that create collocations or TFIDF models reside in this folder.







## Data collection and cleaning
This playground currently uses the following data:
- A dataset with the scripts of the episodes, taken from here: https://www.kaggle.com/gjbroughton/start-trek-scripts
For this part of the data, functions that can extract the episode title from the scripts as well as clean up (remove speakers for example) the text are available.

- A dataset of plot descriptions that I scraped from the Wikipedia articles. The scripts for downloading the files and extracting the actual descriptions of the plots can be found in the data folder.
This part of the data was scraped from Wikipedia. The Plot Descriptions of the episodes were extracted/parsed with a simple script.


## 



