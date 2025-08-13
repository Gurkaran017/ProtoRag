from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from splitter import split
import config

def get_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    chunks = split()
    return FAISS.from_documents(chunks, embeddings)
