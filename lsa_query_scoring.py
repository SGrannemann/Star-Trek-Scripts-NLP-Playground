from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from nltk.tokenize import word_tokenize
from joblib import load
import pandas as pd
import seaborn as sns


# TODO: Adjust the querying to the one used in the tfidf scoring.

bigrams = Phraser.load('bigram_model.pkl')

tng_series_scripts_cleaned = pd.read_pickle('cleaned_tng_scripts.pkl')

tfidf_model = load('tfidf_model.pkl')


tfidf_episodes = tfidf_model.transform(tng_series_scripts_cleaned['text with bigrams'])

tfidf_episodes_frames = pd.DataFrame(tfidf_episodes.toarray())




# convert the vector space to LSA space with reduced dimensions
# we use 160 components as that retains 95% of the variance in the dataset.

svd = TruncatedSVD(n_components=160, random_state=42)
svd.fit(tfidf_episodes)
lsa_episodes = svd.transform(tfidf_episodes)

query = input('What are you looking for?')
bigram_query = [' '.join(bigrams[word_tokenize(query)])]

lsa_query = svd.transform(tfidf_model.transform(bigram_query))

cosineSimilarities = cosine_similarity(lsa_query, lsa_episodes).flatten()
print(cosineSimilarities)
print(len(cosineSimilarities))

# find the highest 5 episodes
search_results = [(index, score) for index,score in enumerate(cosineSimilarities)]
search_results.sort(key= lambda x: x[1], reverse=True)
print(search_results)

