import os
from collections import defaultdict
import gradio as gr
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from config import CHROMA_DIR, EMBED_MODEL, LLM_MODEL

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant. Use only the context below to answer the question.
If the answer isn't in the context, say "I don't have enough information in the provided documents."

Context: {context}"""),
    ("human", "{question}"),
])

llm = ChatOllama(model=LLM_MODEL, temperature=0.2)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_sources(docs):
    source_pages = defaultdict(set)
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", None)
        name = os.path.splitext(os.path.basename(source))[0]
        if page is not None:
            source_pages[name].add(int(page) + 1)
        else:
            source_pages[name] 
    parts = []
    for name, pages in source_pages.items():
        if pages:
            pages_str = ", ".join(f"p{p}" for p in sorted(pages))
            parts.append(f"{name} {pages_str}")
        else:
            parts.append(name)
    return "Sources: " + ", ".join(parts)

def ask(question, history):
    if not question.strip():
        yield "Please enter a question."
        return

    docs = retriever.invoke(question)
    context = format_docs(docs)
    messages = prompt.format_messages(context=context, question=question)

    partial = ""
    for chunk in llm.stream(messages):
        partial += chunk.content
        yield partial

    partial += "\n\n---\n" + format_sources(docs)
    yield partial

gr.ChatInterface(
    ask,
    title="The Study RAGbot",
    description="Ask me anything.",
).launch(theme=gr.themes.Glass())

