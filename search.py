import os
import langchain_ollama
import requests as r
import json
import psycopg2 as pg
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore


def embed_documents():
    conn = pg.connect(os.getenv("PG_CONN_STR"))
    cur = conn.cursor()

    embedding = OllamaEmbeddings(
        model="llama3",
    )
    files = os.listdir("data")
    for file in files:
        conn = pg.connect("dbname=vector_rag user=postgres password=postgres")
        conn.autocommit = True
        cur = conn.cursor()
        print(f"embedding: {file}")
        path = f"data/{file}"
        with open(path, "r") as f:
            document = f.read()

        emb = embedding.embed_query(text=document)

        cur.execute(
            "INSERT INTO laws (name, content, embedding) VALUES (%s, %s, %s)",
            (file, document, emb),
        )
        cur.close()
        conn.close()


if __name__ == "__main__":
    # embed_documents()

    query = input("Hvilken lov vil du gerne søge efter\n")

    emb = OllamaEmbeddings(model="llama3").embed_query(query)
    print("finished embedding")

    conn = pg.connect("dbname=vector_rag user=postgres password=postgres")
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(
        "SELECT name, embedding <-> %s FROM laws ORDER BY embedding <-> %s::vector LIMIT 10;",
        (str(emb), str(emb)),
    )

    result = cur.fetchall()
    print(result)

    cur.close()
    conn.close()
