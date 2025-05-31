from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader ('./faissLoaders/attentionPaper.pdf')

pdfLoaderDocument = loader.load ()

textSplitter = RecursiveCharacterTextSplitter (
    chunk_size = 1000,
    chunk_overlap = 200
)

documents = textSplitter.split_documents (pdfLoaderDocument)

vectorDatabase = FAISS.from_documents (
    documents = documents[:5],
    embedding = OllamaEmbeddings (model='llama3.2:1b')
)

vectorDatabase.save_local ('faissLoaders')