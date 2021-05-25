# Star-Trek-Scripts-NLP-Playground
## Project as a training ground for different aspects of NLP
This project is intended as a toy project to practice different aspects of NLP, from collecting and preprocessing data to the different vector space transformations and different options for scoring document similarity.

## Project structure

This project comes with multiple folders to organize all the files. These folders are:
- Analysis
- Data
- Preprocessing

# Analysis
In this folder, you can find multiple IPython Notebooks for an initial, simple analysis of the data as well as results for optimizing hyperparameters, e.g. optimal number of dimensions for LSA.

# Data
In this folder, you can find the raw data as well as artifacts generated for the different steps of the projects: Serialized DataFrames, collocation models etc.
Scripts for collecting additional data can be found here as well.


# Preprocessing
In this folder, you can find scripts that preprocess the data, either for extracting the text from the original files or for cleaning up text data etc.







## Data collection and cleaning
This playground currently uses the following data:
- A dataset with the scripts of the episodes, taken from here: https://www.kaggle.com/gjbroughton/start-trek-scripts

- A dataset of plot descriptions that I scraped from the Wikipedia articles. The scripts for downloading the files and extracting the actual descriptions of the plots can be found in the data folder.



The _use cases_ that are currently planned are:
- Query options via TFIDF,LSA and word2vec



