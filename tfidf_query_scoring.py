"""Module to try out tfidf based scoring of a query input by the user."""

from gensim.models.phrases import Phraser
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import pandas as pd


bigrams = Phraser.load('bigram_model.pkl')

tng_series_scripts_cleaned = pd.read_pickle('cleaned_tng_scripts.pkl')

tfidf_model = load('tfidf_model.pkl')


tfidf_episodes = tfidf_model.transform(tng_series_scripts_cleaned['text with bigrams'])

tfidf_episodes_frames = pd.DataFrame(tfidf_episodes.toarray())


    
query = input('What are you looking for?')
bigram_query = [' '.join(bigrams[word_tokenize(query)])]

tfidf_query = tfidf_model.transform(bigram_query)

cosineSimilarities = cosine_similarity(tfidf_query, tfidf_episodes).flatten()
print(cosineSimilarities)
print(len(cosineSimilarities))

# get list of all titles (starting from episode 0 ascending)
list_of_titles = list(tng_series_scripts_cleaned['title'].values)
# combine the title with the respective cosine similarity for the query, gives tuples: (title, score)
search_results = list(zip(list_of_titles, cosineSimilarities))

# sort the tuples of (title, score) by score
search_results.sort(key= lambda x: x[1], reverse=True)
print(search_results)