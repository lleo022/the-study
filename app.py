import gradio as gr
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

CHROMA_DIR = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant. Use only the context below to answer the question.
If the answer isn't in the context, say "I don't have enough information in the provided documents."

Context: {context}"""),
    ("human", "{question}"),
])

llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(question, history):
    if not question.strip():
        return "Please enter a question."
    return chain.invoke(question)

gr.ChatInterface(
    ask,
    title="The Study Ragbot",
    description="Ask me anything.",
).launch(theme=gr.themes.Glass())

