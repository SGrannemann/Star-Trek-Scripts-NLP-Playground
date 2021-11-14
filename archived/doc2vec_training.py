"""Script for finding the optimal hyperparameters for the doc2vec approach to the query use case."""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phraser
import spacy
from pathlib import Path
import pandas as pd
from nltk.tokenize import sent_tokenize
from typing import List, Dict
from itertools import product
import json


def get_score(queries_to_results: Dict[str, str], model: Doc2Vec) -> int:
    """function that checks how many test queries return the desired result for the model.
    Score ranges from 0 (for no query the top 3 results included our desired result) to 3 (for each 
    query we found the wanted title in the top 3 results).
    """
    score = 0
    
    for query, desired_result in queries_to_results.items():
        # lemmatize the query
        query = bigrams[[token.lemma_ for token in nlp(query)]]
        # infer the vector from the model - automatically ignores words not contained in the vocab
        query_vector = model.infer_vector(query, steps=10)
        most_similar = model.docvecs.most_similar([query_vector], topn=3)
        results = [scripts_and_plots.iloc[index].Title for index, _ in most_similar]
        if desired_result in results:
            score += 1
    return score

def test_params(epochs: int, min_count: int, vector_size: int, negative: int, sample: float) -> int:
    """ test different parameter combinations"""
    # create the model with the given params
    
    doc2vec_model = Doc2Vec(documents=DOCUMENTS, dm=1, workers=3, hs=0, epochs=epochs, min_count=min_count, vector_size=vector_size, negative=negative, sample=sample)
    # get the score for the model with the selected params
    

    return get_score(queries_to_desired_results, doc2vec_model) 


# get three test queries in order
nlp = spacy.load('en_core_web_sm')
PATH_TO_DATA = Path.cwd() / Path('data')
bigrams = Phraser.load(str(PATH_TO_DATA) + '/bigram_model.pkl')

# these are the test queries
queries = ['Geordi captured by Romulans and mind controlled', 'Picard becomes Borg', 'Q judges humanity']



# for the three test queries, these are the results we would ideally want
GOOD_RESULTS = ["the mind's eye", 'the best of both worlds, part ii', 'encounter at farpoint']



# make a mapping of query: result via dict
queries_to_desired_results = dict(zip(queries, GOOD_RESULTS))



# prepare the documents for Doc2Vec training

# get the complete data to be able to train the model
scripts_and_plots = pd.read_pickle(str(Path.cwd() / Path('data')) + '/scripts_and_plots.pkl')
complete_corpus = list(scripts_and_plots['complete_text_with_bigrams'])

episodes = []
# we need this tokenization -> join loop to ensure a correct split of sentences
# especially regarding lots [] and () in the episode scripts.
for episode in complete_corpus:
    episodes.append(' '.join(sent_tokenize(episode)))

# create tagged documents for doc2vec, we do not need special categories.
DOCUMENTS = [TaggedDocument(doc, [i]) for i, doc in enumerate(episodes)]









# train the model with different param options
params_to_result = {}
epochs_to_test = [10, 20, 50]
min_count_to_test = [1, 3, 5]
vector_size_to_test = [50, 150, 300]
negative_to_test = [5, 10, 15, 20]
sample_to_test = [0, 0.001, 0.00001]   

param_combinations  = list(product(epochs_to_test, min_count_to_test, vector_size_to_test, negative_to_test, sample_to_test))
for param_combination in param_combinations:
    params_to_result[param_combination] = test_params(*param_combination)

print(params_to_result)

with open(Path.cwd() / Path('/doc2vec_results.json'), "w") as outfile: 
    json.dump(params_to_result, outfile)
