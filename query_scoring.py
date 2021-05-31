"""Module to try out tfidf based scoring of a query input by the user."""

from gensim.models.phrases import Phraser
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy


bigrams = Phraser.load('bigram_model.pkl')

scripts_cleaned = pd.read_pickle('scripts_and_plots.pkl')
nlp = spacy.load('en_core_web_sm')

# get tfidf representation
tfidf_model = load('tfidf_model.pkl')
tfidf_episodes = tfidf_model.transform(scripts_cleaned['complete_text_with_bigrams'])

tfidf_episodes_frames = pd.DataFrame(tfidf_episodes.toarray())

# get LSA representation
svd_model = load('svd_model.pkl')
lsa_episodes = svd_model.transform(tfidf_episodes)
    
query = input('What are you looking for?')
bigram_query = [' '.join(bigrams[[token.lemma_ for token in nlp(query)]])]

tfidf_query = tfidf_model.transform(bigram_query)
lsa_query = svd_model.transform(tfidf_query)

cosineSimilarities_tfidf = cosine_similarity(tfidf_query, tfidf_episodes).flatten()
print(cosineSimilarities_tfidf)
print(len(cosineSimilarities_tfidf))

cosineSimilarities_lsa = cosine_similarity(lsa_query, lsa_episodes).flatten()
print(cosineSimilarities_lsa)
print(len(cosineSimilarities_lsa))

# get list of all titles (starting from episode 0 ascending)
list_of_titles = list(scripts_cleaned['Title'].values)
# combine the title with the respective cosine similarity for the query, gives tuples: (title, score)
search_results = list(zip(list_of_titles, cosineSimilarities_tfidf, cosineSimilarities_lsa))

# sort the tuples of (title, score) by score
search_results.sort(key= lambda x: x[1], reverse=True)
print('The search results using TFIDF cosine similarity scoring were:', search_results)

search_results.sort(key= lambda x: x[2], reverse=True)
print('The search results using LSA cosine similarity scoring were:', search_results)