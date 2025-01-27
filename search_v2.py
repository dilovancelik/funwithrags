from langchain import hub
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_postgres import PGVector
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(model="llama3")

model_name = "BAAI/bge-multilingual-gemma2"
model_kwargs = { 'device': 'mps' }
encode_kwargs = { 'normalize_embeddings': False }

embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
#embedding=OllamaEmbeddings(model="llama3")

vectorstore= PGVector(
    embeddings=embedding,
    collection_name="laws_gemma",
    connection=os.getenv("PG_CONN_STR")
)

prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"], k=10)
    print(retrieved_docs)
    return { "context": retrieved_docs }

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)

    return { "answer": response.content }


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


question = input("Hvad vil du vide om loven\n")
response = graph.invoke({ "question": question })
print(response["answer"])

"""
question = input("input\n")
retrieve({"question": question})
"""
