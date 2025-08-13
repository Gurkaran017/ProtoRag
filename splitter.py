from langchain.text_splitter import RecursiveCharacterTextSplitter
from loader import load

def split():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = load()
    # Split the loaded PDF documents into smaller chunks
    split_docs = splitter.split_documents(docs)
    return split_docs
