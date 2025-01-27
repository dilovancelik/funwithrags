from glob import glob
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document
import regex as re
from ollama import chat, ChatResponse
import json

def custom_splitter(folder: str) -> List[Document]:
    documents = []
    files = glob("*.txt", root_dir=folder)
    with tqdm(total=len(files), desc="Splitting documents") as splitbar:
        for file in files[:4]:
            with open(f"{folder}/{file}", "r") as f:
                text = f.read()
                start = 0
                """
                while start < len(text):
                    re_kap = r"\n\s+?Kapitel"
                    re_art = r"\n\s+?Artikel"
                    re_par = r"\n\s+?§"

                    chapter = (
                        re.search(
                            re_kap, text[start : start + 2000], re.IGNORECASE
                        ).end()
                        if re.search(re_kap, text[start : start + 2000], re.IGNORECASE)
                        else 99999
                    )
                    artikel = (
                        re.search(
                            re_art, text[start : start + 2000], re.IGNORECASE
                        ).end()
                        if re.search(re_art, text[start : start + 2000], re.IGNORECASE)
                        else 99999
                    )
                    paragraph = 99999
                    par_matches = [
                        match.span()[1]
                        for match in re.finditer(
                            re_par, text[start : start + 2000], re.IGNORECASE
                        )
                    ]

                    if len(par_matches) > 0:
                        paragraph = par_matches[-1]

                    content = ""
                    if chapter >= 500 & chapter <= 2000:
                        content = text[start : start + chapter]
                        start += chapter
                    elif artikel >= 500 & artikel <= 2000:
                        content = text[start : start + artikel]
                        start += artikel
                    elif paragraph != 99999:
                        content = text[start : start + paragraph]
                        start += paragraph
                    else:
                        content = text[start : start + 2000]
                        start += 1600
                """
                questions = get_questions(context=text)
                for question in questions:
                    documents.append({
                        "question": question,
                        "context": text,
                        "source": file,
                    })
            splitbar.update(1)
    return documents


def get_questions(context: str) -> List[str]:
    content = f"Hvilke spørsmål ville passe godt til følgende tekst, jeg vil have svaret i formattet spørgsmål 1|spørgsmål 2|..| spørgsmål n og uden wrapper tekst: {context}"
    message = {"role": "user", "content": content}
    response: ChatResponse = chat(model="llama3.3", messages=[message])
    while True: 
        if response.message.content.find("|") != -1:
            questions = response.message.content.split("|")
            break
        else:
            message = {"role": "user", "content": f"Jeg vil gerne have skrevet dette om til en række spørgsmål i følgende format 'spørgsmål 1|spørgsmål 2|..| spørgsmål': {response.message.content}"}
            response = chat(model="llama3.3")

    return questions

documents = custom_splitter("data")
with open("split_docs.json", "w") as outfile:
    json.dump(documents, outfile)

