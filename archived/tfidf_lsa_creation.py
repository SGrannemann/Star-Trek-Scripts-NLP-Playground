"""Creates a serialized version of the TFIDF and LSA models for the episodes."""
from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

PATH_TO_DATA = Path.cwd() / Path('data')

scripts_cleaned = pd.read_pickle(PATH_TO_DATA / Path('scripts_and_plots.pkl'))


tfidf_model = TfidfVectorizer().fit(scripts_cleaned['complete_text_with_bigrams'])

dump(tfidf_model, 'tfidf_model.pkl')

tfidf_episodes = tfidf_model.transform(scripts_cleaned['complete_text_with_bigrams'])

tfidf_episodes_frames = pd.DataFrame(tfidf_episodes.toarray())
print(tfidf_episodes_frames.head(10))



# convert the vector space to LSA space with reduced dimensions
# we use 430 components as that retains 95% of the variance in the dataset.
# to try out different dimensions, use the optimal_lsa.py file in the analysis folder.

svd = TruncatedSVD(n_components=430, random_state=42)
svd_model = svd.fit(tfidf_episodes)
dump(svd_model, 'svd_model.pkl')

lsa_episodes = svd_model.transform(tfidf_episodes)

lsa_episodes_frames = pd.DataFrame(lsa_episodes)
print(lsa_episodes_frames.head(10))
