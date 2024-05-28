# function support rag pipeline
from typing import List

# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
import uuid

# from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain_community.document_loaders import TextLoader, DirectoryLoader

import os
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np
from langchain.schema import BaseRetriever, Document
from typing import List
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.vectorstores import VectorStore
from langchain.load import dumps, loads
from typing import Any


def load_data(data_path):
    folders = os.listdir(data_path)
    dir_loaders = []
    loaded_documents = []

    for folder in folders:
        dir_loader = DirectoryLoader(os.path.join(data_path, folder), loader_cls=TextLoader)
        dir_loaders.append(dir_loader)

    for dir_loader in dir_loaders:
        loaded_documents.extend(dir_loader.load())

    return loaded_documents

def process_data(data: List[str], child_text_splitter, embedding, vectorstore_name: str) -> MultiVectorRetriever:

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name=vectorstore_name,
        embedding_function=embedding,
        # collection_metadata={"hnsw:space": "cosine"}
    )

    # The storage layer for the parent documents
    store = InMemoryStore()
    id_key = "doc_id"

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 10}
    )

    doc_ids = [str(uuid.uuid4()) for _ in data]
    sub_docs = []

    for i, doc in enumerate(data):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        sub_docs.extend(_sub_docs)

    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, data)))

    return vectorstore, retriever

class CustomRetriever(BaseRetriever):
    # vectorstores:Chroma
    retriever:Any

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
            and an optional parameter k used in the RRF formula """

        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)   #[:10] #Top 10
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        rr_list=[]
        for doc in reranked_results:
          rr_list.append(doc[0])
        return rr_list


    def _get_relevant_documents(
        self, queries: list, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Use your existing retriever to get the documents
        documents=[]
        for i in range(len(queries)):
          document = self.retriever.get_relevant_documents(queries[i], callbacks=run_manager.get_child())
          documents.append(document)

        unique_documents = self.reciprocal_rank_fusion(documents)

        # Get page content
        docs_content = []
        for i in range(len(unique_documents)):
          docs_content.append(unique_documents[i].page_content)

        # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # model = CrossEncoder('nnngoc/ms-marco-MiniLM-L-6-v2-641M')
        # model = CrossEncoder('nnngoc/ms-marco-MiniLM-L-6-v2-642M-2') *
        # model = CrossEncoder('nnngoc/ms-marco-MiniLM-L-6-v2-644M-1')
        # model = CrossEncoder('nnngoc/ms-marco-MiniLM-L-6-v2-32-2M-2')
        # model = CrossEncoder('nnngoc/ms-marco-MiniLM-L-6-v2-32-5M-1')
        model = CrossEncoder('nnngoc/ms-marco-MiniLM-L-6-v2-32-6M-1')

        # So we create the respective sentence combinations
        sentence_combinations = [[queries[0], document] for document in docs_content]

        # Compute the similarity scores for these combinations
        similarity_scores = model.predict(sentence_combinations)

        # Sort the scores in decreasing order
        sim_scores_argsort = reversed(np.argsort(similarity_scores))

        # Store the rerank document in new list
        docs = []
        for idx in sim_scores_argsort:
          docs.append(unique_documents[idx])

        docs_top_10 = docs[0:10]

        return docs_top_10
    

import cohere
COHERE_API_KEY = 'axMzubIv9l3UTObYnIaHuZhE6tR3Nj8eGReXTws9'

class CustomRetriever1(BaseRetriever):
    # vectorstores:Chroma
    retriever:Any

    def reciprocal_rank_fusion(self, results: list[list], k=60):
        """ Reciprocal_rank_fusion that takes multiple lists of ranked documents
            and an optional parameter k used in the RRF formula """

        # Initialize a dictionary to hold fused scores for each unique document
        fused_scores = {}

        # Iterate through each list of ranked documents
        for docs in results:
            # Iterate through each document in the list, with its rank (position in the list)
            for rank, doc in enumerate(docs):
                # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
                doc_str = dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                # Retrieve the current score of the document, if any
                previous_score = fused_scores[doc_str]
                # Update the score of the document using the RRF formula: 1 / (rank + k)
                fused_scores[doc_str] += 1 / (rank + k)

        # Sort the documents based on their fused scores in descending order to get the final reranked results
        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)   #[:10] #Top 10
        ]

        # Return the reranked results as a list of tuples, each containing the document and its fused score
        rr_list=[]
        for doc in reranked_results:
          rr_list.append(doc[0])
        return rr_list[:30]

    def _get_relevant_documents(
        self, queries: list, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Use your existing retriever to get the documents
        documents=[]
        for i in range(len(queries)):
          document = self.retriever.get_relevant_documents(queries[i], callbacks=run_manager.get_child())
          documents.append(document)

        unique_documents = self.reciprocal_rank_fusion(documents)

        # Get page content
        docs_content = []
        for i in range(len(unique_documents)):
          docs_content.append(unique_documents[i].page_content)

        co = cohere.Client(COHERE_API_KEY)
        results = co.rerank(query=queries[0], documents=docs_content, top_n=5, model='rerank-multilingual-v3.0', return_documents=True)

        reranked_indices = [result.index for result in results.results]

        sorted_documents = [unique_documents[idx] for idx in reranked_indices]

        return sorted_documents