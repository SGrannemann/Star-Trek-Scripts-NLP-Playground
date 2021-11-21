# this script combines a query classification (question yes/no) with either document retrieval or answering a question
from pathlib import Path
from haystack.document_store import InMemoryDocumentStore
from haystack.retriever import DensePassageRetriever
from haystack.reader import FARMReader
#from haystack.query_classifierimport SklearnQueryClassifier # this does not work as of haystack 0.10.0


# get data
documents = []
series_abbrev = ['ds9']#, 'voy', 'tng']
for series in series_abbrev:
    for document_path in (Path.cwd() / Path('data') / Path('scraped') / Path(series) / Path('processed')).glob('*.txt'):
        with open(document_path, 'r', encoding='utf-8') as file:
            documents.append({'text': file.read(), 'meta': {'name': document_path.stem}})

# set up document store (in memory for now) 

doc_store = InMemoryDocumentStore(similarity='dot_product')
doc_store.write_documents(documents)

# set up DPR retriever
retriever = DensePassageRetriever(document_store=doc_store, query_embedding_model='facebook/dpr-question_encoder-single-nq-base',
                                    passage_embedding_model='facebook/dpr-ctx_encoder-single-nq-base')
# the doc store needs to know the embeddings too
doc_store.update_embeddings(retriever)

# get query
query1 = 'Who captured Geordi?'
query2 = 'sisko and jake baseball'

# set up query classifier # left out until the query classifiers are available

# return ranked search results

results = retriever.retrieve(query=query2, top_k=5)
for result in results:
    print(result.meta['name'])

# set up FARMReader for question answering in case of question query