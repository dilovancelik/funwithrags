import glob
from sentence_transformers import SentenceTransformer
import psycopg2 as pg
import regex as re
from tqdm import tqdm

model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
model = SentenceTransformer(model_name)


def embed_documents(speech, document, line, embedding):
    conn = pg.connect("dbname=vector_rag user=postgres password=postgres")
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO speeches_embeddings_v2 (speech, context, line, embedding) VALUES (%s, %s, %s, %s)",
        (speech, document, line, embedding),
    )
    cur.close()
    conn.close()


files = glob.glob("*", root_dir="notebooks/taler")
with tqdm(total=len(files), desc="Processing Files") as fpbar:
    for file_name in files:
        speech_name = file_name.replace(".txt", "")
        with open(f"notebooks/taler/{file_name}", "r") as f:
            raw_lines = f.readlines()
        lines = []
        context_lines = 0
        char_split = False
        for line in raw_lines:
            if line != "\n":
                lines.append(line)
            if re.search("[a-zA-Z]", line) is None and "\n" != line:
                char_split = True
        context_splits = []

        if char_split:
            context = []
            for line in lines:
                if re.search("[a-zA-Z]", line) is None and "\n" != line:
                    context_splits.append(context)
                    context = []
                else:
                    context.append(line)
                    context_lines += 1
            if context != []:
                context_splits.append(context)
        else:
            chunk_size = 10  # group size
            overlap = 2  # overlap size
            context_splits = [
                lines[i : i + chunk_size]
                for i in range(0, len(lines), chunk_size - overlap)
            ]
            context_lines = len(lines)

        with tqdm(total=context_lines, desc="Embedding and Saving context") as pbar:
            for context in context_splits:
                for line in context:
                    embedding = model.encode(line)
                    str_context = "\n".join(context)
                    embed_documents(
                        speech_name, str_context, line, str(embedding.tolist())
                    )
                    pbar.update(1)
        fpbar.update(1)
