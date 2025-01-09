"""
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
"""
from tqdm import tqdm
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, embeddings #, ChatOllama, llms
from langchain_community.document_loaders import DirectoryLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

embedding=OllamaEmbeddings(model="llama3")
vectorstore= PGVector(
    embeddings=embedding,
    collection_name="laws3.3",
    connection="postgresql+psycopg://postgres:postgres@localhost:5432/vector_rag"
)

loader = DirectoryLoader('data', show_progress=True)
documents = loader.load()
print("started splitting")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
splits = text_splitter.split_documents(documents)

print("started vector load")

with tqdm(total = len(splits), desc="Embedding data") as pbar:
    for split in splits:
        vectorstore.add_documents(documents=[split])
        pbar.update(1)


"""
print("creating vector store")
vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="llama3"))
print("finished creating vector store")
retriever = vectorstore.as_retriever()

prompt = hub.pull('rlm/rag-prompt')

def format_docs(documents):
    return "\n\n".join(document.pagecontent for document in documents)

rag_chain = (
    {"context": retriever | format_docs, "questions": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
"""
