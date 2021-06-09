"""Module to try out tfidf and LSA based scoring of a query input by the user."""
from pathlib import Path
from gensim.models.phrases import Phraser
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy


PATH_TO_DATA = Path.cwd() / Path('data')
bigrams = Phraser.load(str(PATH_TO_DATA) + '/bigram_model.pkl')

scripts_cleaned = pd.read_pickle(PATH_TO_DATA / Path('scripts_and_plots.pkl'))
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

# transform the query to the different vector spaces
tfidf_query = tfidf_model.transform(bigram_query)
lsa_query = svd_model.transform(tfidf_query)

# calculate the similarity of the documents with the query
cosineSimilarities_tfidf = cosine_similarity(tfidf_query, tfidf_episodes).flatten()
cosineSimilarities_lsa = cosine_similarity(lsa_query, lsa_episodes).flatten()


# get list of all titles (starting from episode 0 ascending)
list_of_titles = list(scripts_cleaned['Title'].values)
# combine the title with the respective cosine similarity for the query,
# gives tuples: (title, score)
search_results_tfidf = list(zip(list_of_titles, cosineSimilarities_tfidf))
search_results_lsa = list(zip(list_of_titles, cosineSimilarities_lsa))

# sort the tuples of (title, score) by score
search_results_tfidf.sort(key= lambda x: x[1], reverse=True)
# only use the following line if you are interested in all scores
#print('The search results using TFIDF cosine similarity scoring were:', search_results_tfidf)

search_results_lsa.sort(key= lambda x: x[1], reverse=True)
# only use the following line if you are interested in all scores
#print('The search results using LSA cosine similarity scoring were:', search_results_lsa)

# results next to eacher other
result_frame = pd.DataFrame({'TFIDF': [title for title, _ in search_results_tfidf], 'LSA': [title for title, _ in search_results_lsa]})
print(result_frame.head(10))
