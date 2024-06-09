# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from utility import load_data, process_data, CustomRetriever, CustomRetriever1
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


data1 = load_data('raw_data/sv')
data2 = load_data('raw_data/thacsi')
data3 = load_data('raw_data/tiensi')
data = data1 + data2 + data3

# Embedding model
embedding = HuggingFaceEmbeddings(
    model_name="VoVanPhuc/sup-SimCSE-VietNamese-phobert-base",
    model_kwargs={"device": "cpu"}
)

# The splitter to use to create smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

#####################################################################

vectorstore1, retriever1 = process_data(data, child_text_splitter, embedding, "data")
vectorstore2, retriever2 = process_data(data2, child_text_splitter, embedding, "data2")
vectorstore3, retriever3 = process_data(data3, child_text_splitter, embedding, "data3")
vectorstore4, retriever4 = process_data(data1, child_text_splitter,embedding, "data1") # new

##############################################################################

from flashtext import KeywordProcessor
keyword_processor = KeywordProcessor()
# keyword_processor.add_keyword(<unclean name>, <standardised name>)
keyword_processor.add_keyword('thạc sĩ')
keyword_processor.add_keyword('học viên')
keyword_processor.add_keyword('nghiên cứu sinh')
keyword_processor.add_keyword('tiến sĩ')
keyword_processor.add_keyword('sinh viên') # new

################################################################################
"""
import pandas as pd

faq = "raw_data/faq.xlsx"
df = pd.read_excel(faq)
questions = df["question"].tolist()
answers = df["answer"].tolist()

faq_thsi_q = []
faq_thsi_a = []
faq_tsi_q = []
faq_tsi_a = []

for i in range(len(questions)):
  keywords_found = keyword_processor.extract_keywords(questions[i])
  if 'thạc sĩ' in keywords_found or 'học viên' in keywords_found:
    faq_thsi_q.append(questions[i])
    faq_thsi_a.append(answers[i])

  elif 'nghiên cứu sinh' in keywords_found or 'tiến sĩ' in keywords_found:
    faq_tsi_q.append(questions[i])
    faq_tsi_a.append(answers[i])

import uuid
from langchain_core.documents import Document

def add_faq(retriever, vectorstore, questions, answers):
    id_key = "doc_id"

    doc_ids = [str(uuid.uuid4()) for _ in answers]

    question_ = [
        Document(page_content=s, metadata={id_key: doc_ids[i]})
        for i, s in enumerate(questions)
    ]

    answers_ = [ Document(page_content=s) for s in answers]

    retriever.vectorstore.add_documents(question_)

    retriever.docstore.mset(list(zip(doc_ids, answers_)))

# Add FAQ to vectorstore

add_faq(retriever2, vectorstore2, faq_thsi_q, faq_thsi_a)

add_faq(retriever3, vectorstore3, faq_tsi_q, faq_tsi_a)

add_faq(retriever1, vectorstore1, questions, answers)
"""

##################################################################################

ANYSCALE_API_BASE = "credential-1711634141163"
ANYSCALE_API_KEY = "esecret_chitz7splr5ut6vfvqpn72itd3"
ANYSCALE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# ANYSCALE_MODEL_NAME = "meta-llama/Llama-3-8b-chat-hf"
# ANYSCALE_MODEL_NAME = "google/gemma-7b-it"
# ANYSCALE_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
# ANYSCALE_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"

import os

os.environ["ANYSCALE_API_BASE"] = ANYSCALE_API_BASE
os.environ["ANYSCALE_API_KEY"] = ANYSCALE_API_KEY

from langchain.chains import LLMChain
from langchain_community.llms import Anyscale
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatAnyscale

# llm = Anyscale(model_name=ANYSCALE_MODEL_NAME)
llm= ChatAnyscale(model_name=ANYSCALE_MODEL_NAME, temperature=0)

#####################################################################

# GROQ_API_KEY='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'
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
llm_70b_groq = ChatGroq(temperature=0, model_name="Llama3-70b-8192")

####################################################################

# from langchain_openai.llms.azure import AzureOpenAI
# llm_openai = AzureOpenAI(
#     deployment_name="gpt-35-turbo-instruct",
#     # deployment_name="gpt-35-turbo-16k",
#     api_key = 'c90c0e7fb1894a898c56123580a6ee3e',
#     api_version = "2023-09-15-preview",
#     azure_endpoint = "https://bkchatbot.openai.azure.com/",
#     temperature=0.0,
#     max_tokens=500
# )

from langchain_openai import OpenAI
llm_openai = OpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], model_name="gpt-3.5-turbo-instruct")

##########################################################################

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build prompt
from langchain.prompts import PromptTemplate
template ="""
Trả lời câu hỏi dựa trên những quy định được cung cấp.
Tổng hợp thông tin cần thiết và đưa ra câu trả lời cuối cùng.
Nếu câu trả lời nhiều thông tin, trình bày ở dạng gạch đầu dòng.
Câu trả lời nên bắt đầu bằng: "Theo quy định của Trường ĐH Bách Khoa Tp.HCM, ..."
Nếu trong quy văn bản không có thông tin cho câu trả lời, vui lòng thông báo: "Xin lỗi, tôi không có thông tin cho câu hỏi này!"

Quy định: {context}

Câu hỏi: {question}

Câu trả lời:
"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)

#############################################################################

from langchain_core.runnables import RunnableParallel

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | QA_CHAIN_PROMPT
    # | llm
    | llm_70b_groq
    | StrOutputParser()
)

###############################################################################

from langchain.prompts import ChatPromptTemplate

# Multi Query: Different Perspectives
template = """
### Hãy tạo ra thêm các truy vấn tìm kiếm tương đương ngữ nghĩa với một câu hỏi ban đầu.
Kết quả hiển thị dạng list gồm câu hỏi ban đầu và 2 câu hỏi thay thế.

### Câu hỏi ban đầu: {question}
### Kết quả:

"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import ChatOpenAI

generate_queries = (
    prompt_perspectives
    | llm_openai
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

#########################################################################################

from langchain.retrievers import BM25Retriever, EnsembleRetriever

# initialize the bm25 retriever and chroma retriever
bm25_retriever1 = BM25Retriever.from_documents(data, k=10)
ensemble_retriever1 = EnsembleRetriever(retrievers=[bm25_retriever1, retriever1], weights=[0.5, 0.5])

bm25_retriever2 = BM25Retriever.from_documents(data2, k=10)
ensemble_retriever2 = EnsembleRetriever(retrievers=[bm25_retriever2, retriever2], weights=[0.5, 0.5])

bm25_retriever3 = BM25Retriever.from_documents(data3, k=10)
ensemble_retriever3 = EnsembleRetriever(retrievers=[bm25_retriever3, retriever3], weights=[0.5, 0.5])

#new
bm25_retriever4 = BM25Retriever.from_documents(data1, k=10)
ensemble_retriever4 = EnsembleRetriever(retrievers=[bm25_retriever4, retriever4], weights=[0.5, 0.5])

#########################################################################################

custom_retriever1 = CustomRetriever1(retriever = ensemble_retriever1)
custom_retriever2 = CustomRetriever1(retriever = ensemble_retriever2)
custom_retriever3 = CustomRetriever1(retriever = ensemble_retriever3)
custom_retriever4 = CustomRetriever1(retriever = ensemble_retriever4) #new

multiq_chain1 = generate_queries | custom_retriever1
multiq_chain2 = generate_queries | custom_retriever2
multiq_chain3 = generate_queries | custom_retriever3
multiq_chain4 = generate_queries | custom_retriever4 #new

rag_chain_with_source1 = RunnableParallel(
    {"context": multiq_chain1, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

rag_chain_with_source2 = RunnableParallel(
    {"context": multiq_chain2 , "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

rag_chain_with_source3 = RunnableParallel(
    {"context": multiq_chain3, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

#new
rag_chain_with_source4 = RunnableParallel(
    {"context": multiq_chain4, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

############################################################################################

# from flashtext import KeywordProcessor
# keyword_processor = KeywordProcessor()
# # keyword_processor.add_keyword(<unclean name>, <standardised name>)
# keyword_processor.add_keyword('thạc sĩ')
# keyword_processor.add_keyword('học viên')
# keyword_processor.add_keyword('nghiên cứu sinh')
# keyword_processor.add_keyword('tiến sĩ')

################################################################################

#new
rag_chain = [rag_chain_with_source1, rag_chain_with_source2, rag_chain_with_source3, rag_chain_with_source4]

###################################################################################


def rag_(question: str) -> str:

    # Sets for different categories of keywords
    master_keywords = {'thạc sĩ', 'học viên'}
    phd_keywords = {'nghiên cứu sinh', 'tiến sĩ'}
    student_keyword = 'sinh viên'

    keywords_found = keyword_processor.extract_keywords(question)
    
    # Check for combinations
    if len(keywords_found) >= 2:
        # Check if all keywords are within the same category
        if all(keyword in master_keywords for keyword in keywords_found):
            response = rag_chain[1].invoke(question)
        elif all(keyword in phd_keywords for keyword in keywords_found):
            response = rag_chain[2].invoke(question)
        else:
            response = rag_chain[0].invoke(question)

    elif any(keyword in master_keywords for keyword in keywords_found):
        response = rag_chain[1].invoke(question)
    elif any(keyword in phd_keywords for keyword in keywords_found):
        response = rag_chain[2].invoke(question)
    elif student_keyword in keywords_found:
        response = rag_chain[3].invoke(question)
    
    else:
        response = rag_chain[0].invoke(question)

    return response['answer']

# def rag_(question: str) -> str:

#     keywords_found = keyword_processor.extract_keywords(question)
#     if 'thạc sĩ' in keywords_found or 'học viên' in keywords_found:
#       response = rag_chain[1].invoke(question)
#     elif 'nghiên cứu sinh' in keywords_found or 'tiến sĩ' in keywords_found:
#       response = rag_chain[2].invoke(question)
#     else:
#       response = rag_chain[0].invoke(question)
    
#     return response['answer']


###################################################################################


# # Run chain
# from langchain.chains import RetrievalQA

# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        verbose=False,
#                                        # retriever=vectordb.as_retriever(),
#                                        retriever=custom_retriever,
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

# def remove_special_characters(text):
#     text = text.replace('].', '')
#     text = text.replace('/.', '')
#     text = text.replace('/.-', '')
#     text = text.replace('-', '')
#     return text

# def rag(question: str) -> str:
#     # call QA chain
#     response = qa_chain({"query": question})

#     return remove_special_characters(response["result"])

