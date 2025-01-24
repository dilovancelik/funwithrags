import sqlalchemy as sql
import pandas as pd
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

engine = sql.create_engine(os.getenv("PG_CONN_STR"))

input = input("hvad vil du gerne embedde")
emb = OllamaEmbeddings(model="llama3").embed_query(input)


query = f"SELECT cmetadata, document, embedding FROM langchain_pg_embedding  WHERE collection_id = '3555638c-bca3-4d77-86a9-9e3ce7d4f2f0' ORDER BY embedding <-> {emb}::vector LIMIT 10"

df = pd.read_sql(query, engine)
