from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Environment Variables
load_dotenv ()
os.environ['GROQ_API_KEY'] = os.getenv ('GROQ_API_KEY')

# Data Loaders
loader = PyPDFLoader ('attentionPaper.pdf')
pages = loader.load ()

# Text Splitter for RAG
text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=100)

# Chunks for processing
chunks = text_splitter.split_documents (pages)
print (f'Split {len (pages)} pages into {len (chunks)} chunks')

# Large Language Model
llm = ChatGroq (model="llama3-8b-8192")

# RAG based summerizer

# Stuff style RAG Summarizer
# summarization_chain = load_summarize_chain (llm=llm, chain_type='stuff')
# result = summarization_chain.invoke (chunks[:-30])
# result['output_text']

# Map Reduce style RAG Summarizer
summarization_chain = load_summarize_chain (llm=llm, chain_type='map_reduce')
result = summarization_chain.invoke (chunks)
print (result['output_text'])