"""Creates a serialized version of the TFIDF model for the TNG episodes."""
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


scripts_cleaned = pd.read_pickle('scripts_and_plots.pkl')


tfidf_model = TfidfVectorizer().fit(scripts_cleaned['complete_text_with_bigrams'])

dump(tfidf_model, 'tfidf_model.pkl')

tfidf_episodes = tfidf_model.transform(scripts_cleaned['complete_text_with_bigrams'])

#tfidf_episodes_frames = pd.DataFrame(tfidf_episodes.toarray())
#print(tfidf_episodes_frames.head(10))



# convert the vector space to LSA space with reduced dimensions
# we use 160 components as that retains 95% of the variance in the dataset.

svd = TruncatedSVD(n_components=160, random_state=42)
svd_model = svd.fit(tfidf_episodes)
dump(svd_model, 'svd_model.pkl')

