import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

load_dotenv ()

groq_api_key = os.environ['GROQ_API_KEY']

if 'vector' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings (model = 'llama3.2:1b')
    st.session_state.loader = WebBaseLoader ('https://docs.smith.langchain.com/')
    st.session_state.docs = st.session_state.loader.load ()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter (chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_document = st.session_state.text_splitter.split_documents (st.session_state.docs[:5])
    st.session_state.database = FAISS.from_documents (st.session_state.final_document, st.session_state.embeddings)

st.title ('Generic GROQ Trial')
llm = ChatGroq (groq_api_key=groq_api_key, model='llama3-8b-8192')

prompt = ChatPromptTemplate.from_template (
    '''
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
            {context}
        </context>
        Questions : {input}
    '''
)

document_chain = create_stuff_documents_chain (llm, prompt)
retriever = st.session_state.database.as_retriever ()
retriever_chain = create_retrieval_chain (retriever, document_chain)

prompt = st.text_input ('Input your prompt here')

if prompt:
    response = retriever_chain.invoke ({
        'input' : prompt
    })
    st.write (response['answer'])