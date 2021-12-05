'''Script that sets up document stores from the files created in this project.
Lets you search for relevant episodes from TNG, VOY, and DS9 with either a TfIdf
based search or a dense vector space search using dense passage retrieval.
For details please see the Haystack documentation.
Last part is a simple question answering system.
'''

from pathlib import Path
from haystack.document_store import InMemoryDocumentStore
from haystack.retriever import DensePassageRetriever
from haystack.retriever.sparse import TfidfRetriever
from haystack.reader import FARMReader
from haystack.pipeline import ExtractiveQAPipeline
# from haystack.query_classifierimport SklearnQueryClassifier # this does not work as of haystack 0.10.0


# get data
documents = []
series_abbrev = ['tng', 'voy', 'tng']
for series in series_abbrev:
    for document_path in (Path.cwd() / Path('data') / Path('scraped') / Path(series) / Path('processed')).glob('*.txt'):
        with open(document_path, 'r', encoding='utf-8') as file:
            documents.append({'text': file.read(), 'meta': {'name': document_path.stem}})

# set up document store (in memory for now)

DPR_doc_store = InMemoryDocumentStore(similarity='dot_product')
DPR_doc_store.write_documents(documents)


TfIdf_doc_store = InMemoryDocumentStore()
TfIdf_doc_store.write_documents(documents)

# set up DPR retriever
DPR_retriever = DensePassageRetriever(document_store=DPR_doc_store,
                                      query_embedding_model='facebook/dpr-question_encoder-single-nq-base',
                                      passage_embedding_model='facebook/dpr-ctx_encoder-single-nq-base')

# the doc store needs to know the embeddings too
DPR_doc_store.update_embeddings(DPR_retriever)


TfIdf_retriever = TfidfRetriever(TfIdf_doc_store)

# some test queries
queries = ['Geordi captured by Romulans and mindcontrolled', 'Picard becomes Borg', 'Q judges humanity']

# return search results
for query in queries:
    results_DPR = DPR_retriever.retrieve(query=query, top_k=3)
    results_tfidf = TfIdf_retriever.retrieve(query=query, top_k=3)
    for result in results_DPR:
        print(f"DPR Result for {query}:  {result.meta['name']}")
    for result in results_tfidf:
        print(f"TfIdf Result for {query}:  {result.meta['name']}")


# set up query classifier # left out until the query classifiers are available


# set up FARMReader for question answering
MODEL = 'deepset/roberta-base-squad2'
reader = FARMReader(MODEL)

# use pipeline with both the DPR retriever and the TfIdf retriever.
# TfIdf returns better search results for the queries I tested (and consequently is better at answering questions)
pipeline_dpr = ExtractiveQAPipeline(reader, DPR_retriever)
pipeline_tfidf = ExtractiveQAPipeline(reader, TfIdf_retriever)
questions = ['Who captures Geordi La Forge?', 'Which tea does Picard drink?', "Who is Data's brother?",
             'What is the name of Benjamin Siskos son?', 'Who is the captain of the USS Voyager?',
             "What is Worf's race?", 'Who is the chief engineer on the USS Enterprise?',
             'What is gold pressed latinum?', 'What type of torpedos does the Enterprise have?']

# simply print the 3 best answers for each question
for question in questions:
    predictions = pipeline_tfidf.run(query=question, params={'Retriever': {'top_k': 5}, 'Reader': {'top_k': 3}})
    for prediction in predictions['answers']:
        print(f"{question}: {prediction['answer']}")
