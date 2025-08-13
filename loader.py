from langchain_community.document_loaders import PyPDFLoader

def load():
    loader = PyPDFLoader('dl-curriculum.pdf')
    docs = loader.load()
    print(len(docs))
    return docs
