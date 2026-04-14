import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

PDF_DIR = "./pdfs"
CHROMA_DIR = "./chroma_db"

all_docs = []
for file in os.listdir(PDF_DIR):
    if file.endswith(".pdf"):
        print(f"Loading: {file}")
        loader = PyPDFLoader(os.path.join(PDF_DIR, file))
        all_docs.extend(loader.load())

print(f"Total pages loaded: {len(all_docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=CHROMA_DIR
)
print(f"Done. {len(chunks)} chunks stored in {CHROMA_DIR}")