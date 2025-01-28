from glob import glob
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document
import regex as re
from ollama import chat, ChatResponse
import json

def get_questions(context: str) -> List[str]:
    content = f"Hvilke danske spørsmål ville passe godt til følgende tekst, jeg vil have svaret i formattet spørgsmål 1|spørgsmål 2|..| spørgsmål n og uden wrapper tekst: {context}"
    message = {"role": "user", "content": content}
    response: ChatResponse = chat(model="llama3.3", messages=[message])
    while True: 
        if response.message.content.find("|") != -1:
            questions = response.message.content.split("|")
            break
        else:
            message = {"role": "user", "content": f"Jeg vil gerne have skrevet dette om til en række spørgsmål i følgende format 'spørgsmål 1|spørgsmål 2|..| spørgsmål på dansk': {response.message.content}"}
            response = chat(model="llama3.3", messages=[message])
            print(response)
    return questions

with open("documents.csv", "r") as f:
    documents = f.readlines()

for doc in documents:
    questions = get_questions(doc)
    with open("results.csv", "a") as res:
        for question in questions:
            obj = { "question": question, "context": doc }
            res.write(f"{json.dumps(obj)}\n")

    

