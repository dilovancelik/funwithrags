from langchain import hub
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOllama(model="llama3")

embedding=OllamaEmbeddings(model="llama3")
vectorstore= PGVector(
    embeddings=embedding,
    collection_name="laws3_split_v2",
    connection=os.getenv("PG_CONN_STR")
)

prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"], k=4)
    for doc in retrieved_docs:
        print(doc)
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
