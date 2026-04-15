# Retrieval Test
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_DIR, EMBED_MODEL

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

query = "Where do we submit?" 
results = db.similarity_search(query, k=3)

print(db._collection.count())
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Source: {doc.metadata.get('source', 'unknown')}, Page: {doc.metadata.get('page', '?')}")
    print(doc.page_content)