import psycopg2 as pg
from sentence_transformers import SentenceTransformer, models
from llama_index.llms.ollama import Ollama

METTE_PROMPT_TEMPLATE = """\
Du er en LLM som giver svarer på hvad Mette Frederiksen syntes om: {question}.

Du må kun besvarer baseret af de nedenstående citater \
Du skal svarer på dansk \

---------------------
{context_str}
---------------------
"""

models.Transformer

if __name__ == "__main__":
    print("Forbereder Mette bot")

    model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    model = SentenceTransformer(model_name)

    llm = Ollama(model="phi4", request_timeout=90)
    while True:
        query = input("Hvilket emne vil du høre Mette Frederiksens mening om?\n")
        print("Tænker ...")
        emb = model.encode(query).tolist()

        print("Henter citater...")
        conn = pg.connect("dbname=vector_rag user=postgres password=postgres")
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute(
            "SELECT speech, context, line FROM speeches_embeddings_v2 ORDER BY embedding <-> %s::vector LIMIT 30;",
            (str(emb),),
        )

        result = cur.fetchall()

        context_all = []
        speeches = []
        for row in result:
            speech = row[0]
            speeches.append(speech)
            context_all.append(f"'{row[1]}'")
            line = row[2]

        prompt = METTE_PROMPT_TEMPLATE.format(
            context_str="\n\n".join(list(set(context_all))), question=query
        )

        cur.close()
        conn.close()

        # print("Prøver at formulere mig...")
        # res = llm.complete(prompt)
        # print(res)
        print("\n\nLink til Taler og Citater som er valgt:")
        for row in result:
            print(
                f"Citat: '{row[2].strip()}'\t\nLink: https://www.dansketaler.dk/tale/{row[0]}\n"
            )
