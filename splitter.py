# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from loader import load

# def split():
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50
#     )
#     docs = load()
#     # Split the loaded PDF documents into smaller chunks
#     split_docs = splitter.split_documents(docs)
#     return split_docs

from loader import load
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Use Gemini embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# SemanticChunker with percentile threshold
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)

def split():
    docs = load()
    # Split the loaded PDF documents into semantic chunks
    split_docs = splitter.split_documents(docs)
    return split_docs
