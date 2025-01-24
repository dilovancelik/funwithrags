"""
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
"""
import os
from dotenv import load_dotenv
from glob import glob
from typing import List
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
from langchain_postgres import PGVector
import regex as re

load_dotenv()

def custom_splitter(folder: str) -> List[Document]:
    documents = []
    files = glob("*.txt", root_dir=folder)
    with tqdm(total=len(files), desc="Splitting documents") as splitbar:
        for file in files:
            with open(f"{folder}/{file}", 'r') as f:
                text = f.read()
                start = 0
                while start < len(text):
                    re_kap = r"\n\s+?Kapitel"
                    re_art = r"\n\s+?Artikel"
                    re_par = r"\n\s+?§"
                    chapter = re.search(re_kap, text[start:start+1000], re.IGNORECASE).end() if re.search(re_kap, text[start:start+1000], re.IGNORECASE) else 99999
                    artikel = re.search(re_art, text[start:start+1000], re.IGNORECASE).end() if re.search(re_art, text[start:start+1000], re.IGNORECASE) else 99999
                    paragraph = 99999
                    par_matches = [match.span()[1] for match in re.finditer(re_par, text[start:start+1000], re.IGNORECASE)]
                    
                    if len(par_matches) > 0:
                        paragraph = par_matches[-1]

                    if chapter != 99999:
                        documents.append(Document(page_content=text[start:start+chapter], metadata={'source': file}))
                        start += chapter
                    elif artikel != 99999:
                        documents.append(Document(page_content=text[start:start+artikel], metadata={'source': file}))
                        start += artikel
                            
                    elif paragraph != 99999:
                        documents.append(Document(page_content=text[start:start+paragraph], metadata={'source': file}))
                        start += paragraph
                    else:
                        documents.append(Document(page_content=text[start:start+1000], metadata={'source': file}))
                        start += 800
            splitbar.update(1)
    return documents

model_name = "BAAI/bge-multilingual-gemma2"
model_kwargs = { 'device': 'mps' }
encode_kwargs = { 'normalize_embeddings': False }

embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

vectorstore = PGVector(
    embeddings=embedding,
    collection_name="laws_gemma",
    connection=os.getenv("PG_CONN_STR")
)

splits = custom_splitter('data2')

with tqdm(total = len(splits), desc="Embedding data") as pbar:
    for split in splits:
        vectorstore.add_documents(documents=[split])
        pbar.update(1)
