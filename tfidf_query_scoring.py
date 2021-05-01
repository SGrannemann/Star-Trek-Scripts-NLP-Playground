from gensim.models.phrases import Phraser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import pandas as pd


def remove_speakers_and_empty_lines(episode_content: str) -> str:
    """Removes superfluous empty lines and the names of the speakers from the input data:
    e.g. :
    Picard: Make it so.
    becomes:
    Make it so.
    """

    cleaned_lines = []
    for line in episode_content.split('\n'):
        # ignore empty lines
        if line == '':
            continue
        # lines that start with square brackets are just information about the location.
        if line.startswith('['):
            continue
        # the actual talking lines always contain a ':' - we will just keep the text, not the talker
        # for this application
        if ':' in line:
            part_to_keep = line.split(':')[1]
            cleaned_lines.append(part_to_keep.strip() + ' \n')
            continue
        # after this string there are only information about the franchise, we can leave those out.
        if line == '<Back':
            break
        
        
        cleaned_lines.append(line.strip() + ' \n')
    return ''.join(cleaned_lines)


if __name__ == '__main__':

    all_series_scripts = pd.read_json('all_scripts_raw.json')
    #  remove the names of the speakers and get rid of the empty lines
    # and I'll focus on The Next Generation Episodes for now
    tng_series_scripts_cleaned = all_series_scripts.TNG.map(remove_speakers_and_empty_lines)
    bigrams = Phraser.load('bigram_model.pkl')
    
    tng_series_scripts_cleaned['text with bigrams'] =  [' '.join(bigrams[word_tokenize(episode_text)]) for episode_text in tng_series_scripts_cleaned]
   
    tfidf_model = TfidfVectorizer().fit(tng_series_scripts_cleaned['text with bigrams'])
    tfidf_episodes = tfidf_model.transform(tng_series_scripts_cleaned['text with bigrams'])

    tfidf_episodes_frames = pd.DataFrame(tfidf_episodes.toarray())
    
    #print(tfidf_episodes)
     
    query = input('What are you looking for?')
    bigram_query = [' '.join(bigrams[word_tokenize(query)])]

    tfidf_query = tfidf_model.transform(bigram_query)

    cosineSimilarities = cosine_similarity(tfidf_query, tfidf_episodes).flatten()
    print(cosineSimilarities)
    print(len(cosineSimilarities))

