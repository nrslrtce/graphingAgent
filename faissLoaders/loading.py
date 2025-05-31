from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv ()
groq_api_key = os.environ['GROQ_API_KEY']

loadingDatabase = FAISS.load_local (
    'faissLoaders',
    OllamaEmbeddings (model='llama3.2:1b'),
    allow_dangerous_deserialization = True
)

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

llm = ChatGroq (groq_api_key=groq_api_key, model='llama3-8b-8192')
documentChain = create_stuff_documents_chain (llm=llm, prompt=prompt)
retriever = loadingDatabase.as_retriever ()
retrieverChain = create_retrieval_chain (retriever, documentChain)
response = retrieverChain.invoke ({'input' : 'What is attention ?'})
print (response['answer'])
