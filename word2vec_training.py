""" Script for finding the optimal hyperparameters for the query use for a word2vec model."""

from gensim.models import Word2Vec
from pathlib import Path
from gensim.models.phrases import Phraser
import spacy
from pathlib import Path
import pandas as pd
from nltk.tokenize import sent_tokenize
from typing import List, Dict
from itertools import product
import multiprocessing
import json

def check_query_vocab(query: List, wordvectors) -> List:
    """Function that checks if a token is contained in the dictionary of the word2vec model.
    If it is not, checks different spellings. If the token is still not found, it is discarded."""
    tokens = query[0].split()
    tokens_to_return = []
    # check for different spelling options of the query and only use the word
    # if any of the spellings are actually in the vocabular
    for token in tokens:
        
        if token in wordvectors:
            tokens_to_return.append(token)
        elif token.lower() in wordvectors:
            tokens_to_return.append(token.lower())
        elif token.title() in wordvectors:
            tokens_to_return.append(token.title())
    return tokens_to_return 



def get_score(queries_to_results: Dict[str, str], model: Word2Vec) -> int:
    """function that checks how many test queries return the desired result for the model.
    For the queries, the score range from 0 to 3."""
    score = 0
    list_of_titles = list(scripts_and_plots['Title'].values)
    
    

    for query, desired_result in queries_to_results.items():
        word2vec_results = []
        bigram_query = [' '.join(bigrams[[token.lemma_ for token in nlp(query)]])]
        bigram_query_vector_checked = check_query_vocab(bigram_query, model.wv)
        for episode in episodes_prepared:
            result = model.wv.n_similarity(bigram_query_vector_checked, episode.split())   
            word2vec_results.append(result)
        search_results_word2vec = list(zip(list_of_titles, word2vec_results))
        search_results_word2vec.sort(key= lambda x: x[1], reverse=True)

        for title, _ in search_results_word2vec[:3]:
            if desired_result in title: 
                score += 1

        
    return score

def test_params(epochs: int, vector_size: int, negative: int, sample: float) -> int:
    """ test different parameter combinations for the word2vec model."""
    # create the model with the given params
    
    word2vec_model = Word2Vec(corpus_file=str(Path.cwd() / Path('data')) + '/star_trek_corpus.txt', min_count=1, iter=epochs, size=vector_size, sg=1, hs=0, workers=multiprocessing.cpu_count(), negative=negative, sample=sample)
    # get the score for the model with the selected params
    

    return get_score(queries_to_desired_results, word2vec_model) 


# get three test queries 
nlp = spacy.load('en_core_web_sm')
PATH_TO_DATA = Path.cwd() / Path('data')
bigrams = Phraser.load(str(PATH_TO_DATA) + '/bigram_model.pkl')

# these are the test queries
queries = ['Geordi captured by Romulans and mindcontrolled', 'Picard becomes Borg', 'Q judges humanity']


# for the three test queries, these are the results we would ideally want
GOOD_RESULTS = ["the mind's eye", 'the best of both worlds, part ii', 'encounter at farpoint']



# make a mapping of query: result via dict
queries_to_desired_results = dict(zip(queries, GOOD_RESULTS))



# prepare the documents for Doc2Vec training

# get the complete data to be able to train the model
scripts_and_plots = pd.read_pickle(str(Path.cwd() / Path('data')) + '/scripts_and_plots.pkl')
complete_corpus = list(scripts_and_plots['complete_text_with_bigrams'])
episodes_prepared = [' '.join(sent_tokenize(episode)) for episode in scripts_and_plots['complete_text_with_bigrams']]

# train the model with different param options
params_to_result = {}
epochs_to_test = [10, 20, 50]
vector_size_to_test = [50, 150, 300]
negative_to_test = [5, 10, 15, 20]
sample_to_test = [0, 0.001, 0.00001]   

# now check each paramter combination
param_combinations  = list(product(epochs_to_test, vector_size_to_test, negative_to_test, sample_to_test))
for param_combination in param_combinations:
    params_to_result[param_combination] = test_params(*param_combination)


# save the dictionary with the results as a json file for later.
with open(Path.cwd() / Path('/word2vec_results.json'), "w") as outfile: 
    json.dump(params_to_result, outfile)

    