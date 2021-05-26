"""Creates a serialized version of the TFIDF model for the TNG episodes."""
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer


tng_series_scripts_cleaned = pd.read_pickle('cleaned_tng_scripts.pkl')


tfidf_model = TfidfVectorizer().fit(tng_series_scripts_cleaned['complete_text_with_bigrams'])

dump(tfidf_model, 'tfidf_model.pkl')

tfidf_episodes = tfidf_model.transform(tng_series_scripts_cleaned['complete_text_with_bigrams'])

tfidf_episodes_frames = pd.DataFrame(tfidf_episodes.toarray())
print(tfidf_episodes_frames.head(10))